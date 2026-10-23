#!/usr/bin/env python3
"""Import an externally quantized TFLite model into a Coral ``coral.graph`` module.

This is the externally-quantized import path: take a real, fully-quantized
TFLite network (for example the pretrained
``mobilenet_v2_1.0_224_quant.tflite``), read its graph
*and its quantization metadata*, and re-materialise it as an MLIR ``func.func``
carrying ``coral.operation = "graph"`` so the native IREE Coral pipeline can
compile it.

Quantization import
-------------------
The source model stores its weights as integers with a per-tensor
(de)quantization: ``real = (q - zero_point) * scale``.  The Coral pipeline
transports weights as INT8 and applies an optional per-node ``weight_scale``
(scalar) / ``bias_scale``, so this tool:

  * imports each weight tensor *losslessly* as INT8.  An asymmetric UINT8 source
    (``v in [0, 255]``) becomes ``q = v - 128`` with ``weight_zero_point = z - 128``
    and the original ``weight_scale``; because ``(q - z') == (v - z)`` the real
    weights are reproduced exactly, with no re-quantisation error,
  * emits the INT32 bias together with ``bias_scale = input.scale * weight.scale``.

The older behaviour -- re-quantising onto a symmetric INT8 grid
``g = max|W| / 127`` -- silently perturbed every filter and is gone.

Per-channel weight quantization (multiple scales per weight tensor) is rejected
with a clear message: the current pipeline transports a single scale per node.

Usage:
    import_tflite.py MODEL.tflite -o OUT.mlir [--tag TAG]
"""

import argparse
import os
import sys

import numpy as np

from tensorflow.lite.python import schema_py_generated as schema

# TFLite builtin operator codes we understand.
BUILTIN = {
    0: "ADD",
    1: "AVERAGE_POOL_2D",
    2: "CONCATENATION",
    3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D",
    9: "FULLY_CONNECTED",
    17: "MAX_POOL_2D",
    22: "RESHAPE",
    25: "SOFTMAX",
}

PADDING = {0: "SAME", 1: "VALID"}
FUSED = {0: None, 1: "relu", 2: "relu", 3: "relu6", 4: "tanh"}

DEFAULT_QUANT_ATTRS = (
    "coral.input.scale = 0.01 : f32, coral.input.zero_point = -128 : i32, "
    "coral.weight.scale = 0.01 : f32, coral.weight.zero_point = 0 : i32, "
    "coral.bias.scale = 0.0001 : f32, coral.bias.zero_point = 0 : i32, "
    "coral.output.scale = 0.01 : f32, coral.output.zero_point = -128 : i32"
)


def interface_quant_attrs(in_tensor, out_tensor):
    """The *source* model's real input/output quantization, as MLIR attributes.

    The Coral pipeline computes every activation scale itself, but the two
    interface tensors are free parameters: the generated model must accept the
    same byte stream the source model accepts.  Emitting the source scale and
    zero point here lets the generator reconstitute the interface tensors
    exactly instead of letting the TFLite converter invent a range (which is
    what produced an input scale of 2/255 instead of the source's 1/128).
    """
    iscale, izp = one_scale(in_tensor)
    oscale, ozp = one_scale(out_tensor)
    itype = tensor_dtype(in_tensor)
    otype = tensor_dtype(out_tensor)
    # The interface travels in the INT8 domain; an asymmetric UINT8 source is
    # shifted by 128 exactly like the weights, so the byte stream is unchanged.
    if itype == 3:
        izp -= 128
    if otype == 3:
        ozp -= 128
    # Keep the (informational) weight/bias scales self-consistent with the real
    # input scale; the per-node values that actually drive generation live in
    # each node's own weight_scale / bias_scale.
    return (
        "coral.input.scale = %s : f32, coral.input.zero_point = %d : i32, "
        "coral.weight.scale = 0.01 : f32, coral.weight.zero_point = 0 : i32, "
        "coral.bias.scale = %s : f32, coral.bias.zero_point = 0 : i32, "
        "coral.output.scale = %s : f32, coral.output.zero_point = %d : i32"
        % (fnum(iscale), izp, fnum(iscale * 0.01), fnum(oscale), ozp)
    )


class ImportError_(Exception):
    pass


def builtin_code(model, op):
    codes = model.OperatorCodes(op.OpcodeIndex())
    bc = codes.BuiltinCode()
    if bc == 0 and codes.DeprecatedBuiltinCode() != 0:
        bc = codes.DeprecatedBuiltinCode()
    return bc


def opts(op, cls):
    """Decode an operator's BuiltinOptions into its typed flatbuffer class."""
    table = op.BuiltinOptions()
    o = cls()
    o.Init(table.Bytes, table.Pos)
    return o


def name_of(subgraph, index):
    t = subgraph.Tensors(index)
    return (t.Name() or b"").decode("utf-8", "replace")


def infer_activation(subgraph, out_index, fused):
    """Recover an activation that the quantizer folded into a tensor's range.

    Older "range quantized" models (such as the pre-1.0 quantized MobileNets)
    leave ``FusedActivationFunction`` at NONE and instead encode RELU/RELU6 in
    the *quantization parameters* of the operator's output tensor: the output
    range starts at exactly 0, so the requantizing saturate that TFLite performs
    when writing the activation clips negative values away, and the upper bound
    (6.0 for RELU6) caps the rest.  Dropping that range silently turns every
    such layer into a linear one, which lets activations grow without bound.

    A non-negative range on its own is *not* enough - an average pool that
    happens to observe only non-negative values also ends in [0, 6] - so the
    tensor name has to name the activation, and the range is then used only as a
    consistency check.  Requiring the name keeps a plain pool/linear output
    linear, which matters because the Edge TPU cannot fuse an activation onto a
    pooling operator.
    """
    if fused is not None:
        return fused
    name = name_of(subgraph, out_index).lower()
    if "relu6" in name:
        candidate = "relu6"
    elif "relu" in name:
        candidate = "relu"
    else:
        return None
    t = subgraph.Tensors(out_index)
    q = t.Quantization()
    if q.ScaleLength() != 1 or t.Type() not in (3, 9):  # uint8 / int8
        return None
    scale = float(q.Scale(0))
    zp = int(q.ZeroPoint(0))
    qmin = 0 if t.Type() == 3 else -128
    if scale <= 0.0 or abs((qmin - zp) * scale) > 1e-6:
        return None  # the range admits negative values -> activation was not fused
    return candidate


def tensor_dtype(t):
    return t.Type()


def read_buffer(model, index):
    t = model.Subgraphs(0).Tensors(index)
    buf = model.Buffers(t.Buffer())
    if buf.DataLength() == 0:
        return None, t
    raw = bytes(buf.Data(j) for j in range(buf.DataLength()))
    np_dtype = {0: np.float32, 2: np.int32, 3: np.uint8, 9: np.int8}[t.Type()]
    return np.frombuffer(raw, dtype=np_dtype), t


def one_scale(t):
    q = t.Quantization()
    n = q.ScaleLength()
    if n == 0:
        raise ImportError_("tensor is not quantized")
    if n != 1:
        raise ImportError_(
            "tensor has %d quantization scales (per-channel weights): the Coral "
            "pipeline transports a single scalar per-node 'weight_scale', so "
            "per-channel models cannot be imported faithfully yet" % n
        )
    return float(q.Scale(0)), int(q.ZeroPoint(0))


def import_weights(model, tensor_index, tensor):
    """Losslessly import a per-tensor quantized weight tensor.

    Returns ``(int8_values, weight_scale, weight_zero_point)`` with each value in
    the INT8 domain.  The source model stores ``real = (v - z) * s``; representing
    the same values as INT8 is a pure zero-point shift::

        q  = v - 128          (UINT8 source; q in [-128, 127])
        z' = z - 128

    so ``(q - z') == (v - z)`` and the real weights are reproduced *exactly* with
    the original scale.  An INT8 source (``v in [-128, 127]``) is already INT8 and
    is passed through unchanged.  The previous implementation instead re-quantised
    onto a symmetric grid ``g = max|W| / 127`` and silently lost precision on
    every filter.
    """
    data, t = read_buffer(model, tensor_index)
    if data is None:
        raise ImportError_("weight tensor '%s' has no inline data"
                           % name_of(model.Subgraphs(0), tensor_index))
    scale, zp = one_scale(t)
    ttype = tensor_dtype(t)
    if ttype == 3:  # UINT8 source
        values = data.astype(np.int16) - 128
        zero_point = zp - 128
    elif ttype == 9:  # INT8 source
        values = data.astype(np.int16)
        zero_point = zp
    else:
        raise ImportError_("weight tensor '%s' has unsupported type %d"
                           % (name_of(model.Subgraphs(0), tensor_index), ttype))
    if values.min() < -128 or values.max() > 127 or not -128 <= zero_point <= 127:
        raise ImportError_(
            "weight tensor '%s' cannot be represented losslessly as INT8"
            % name_of(model.Subgraphs(0), tensor_index))
    return values.astype(np.int8), scale, zero_point


def dequant_bias_scale(model, conv_input_index, weight_index):
    sg = model.Subgraphs(0)
    in_scale, _ = one_scale(sg.Tensors(conv_input_index))
    w_scale, _ = one_scale(sg.Tensors(weight_index))
    return in_scale * w_scale


# --- MLIR formatting --------------------------------------------------------

def nested(shape, flat):
    it = iter(flat)

    def build(dims):
        if len(dims) == 1:
            return "[" + ", ".join(str(next(it)) for _ in range(dims[0])) + "]"
        return "[" + ", ".join(build(dims[1:]) for _ in range(dims[0])) + "]"

    return build(shape)


def dense(values, shape, elem):
    flat = [int(v) for v in np.asarray(values).reshape(-1)]
    shp = "x".join(str(int(d)) for d in shape)
    return "dense<%s> : tensor<%sx%s>" % (nested(list(shape), flat), shp, elem)


def tensor_ty(shape, elem="i8"):
    return "tensor<" + "x".join(str(int(d)) for d in shape) + "x" + elem + ">"


def fnum(x):
    return repr(float(x))


class Builder:
    def __init__(self):
        self.nodes = []
        self.used = {}

    def unique(self, base):
        base = "".join(c if c.isalnum() else "_" for c in base).strip("_") or "t"
        if len(base) > 40:
            base = base[:40]
        if base not in self.used:
            self.used[base] = 0
            return base
        self.used[base] += 1
        return "%s_%d" % (base, self.used[base])

    def add(self, nid, op, inputs, extra="", out_quant=""):
        self.nodes.append(
            '{id = "%s", op = "%s", inputs = [%s]%s%s}'
            % (nid, op, ", ".join('"%s"' % i for i in inputs), extra, out_quant)
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--tag", default="mobilenet")
    ap.add_argument("--func-name", default="run")
    args = ap.parse_args()

    raw = open(args.model, "rb").read()
    model = schema.Model.GetRootAsModel(raw, 0)
    sg = model.Subgraphs(0)

    # tensor index -> coral.graph tensor name
    names = {}
    in_idx = sg.Inputs(0)
    out_idx = sg.Outputs(0)
    in_shape = [int(sg.Tensors(in_idx).Shape(i)) for i in range(sg.Tensors(in_idx).ShapeLength())]
    out_shape = [int(sg.Tensors(out_idx).Shape(i)) for i in range(sg.Tensors(out_idx).ShapeLength())]
    names[in_idx] = "input"

    b = Builder()
    for oi in range(sg.OperatorsLength()):
        op = sg.Operators(oi)
        code = builtin_code(model, op)
        kind = BUILTIN.get(code)
        if kind is None:
            raise ImportError_(
                "operator #%d (%s) is not supported by the Coral importer"
                % (oi, name_of(sg, op.Outputs(0)))
            )
        ins = [op.Inputs(j) for j in range(op.InputsLength())]
        outs = [op.Outputs(j) for j in range(op.OutputsLength())]
        if len(outs) != 1:
            raise ImportError_("operator %s has %d outputs (expected 1)"
                               % (kind, len(outs)))
        out = outs[0]
        nid = b.unique(name_of(sg, out))
        in_names = [names[i] for i in ins if i in names]
        extra = ""
        fused = None

        # The source model's own output (activation) quantization for this
        # operator.  Carrying it through lets the generator reproduce the
        # source's requantization exactly instead of re-deriving activation
        # ranges by calibration, which is what made the deep layers drift.
        oscale, ozp = one_scale(sg.Tensors(out))
        if tensor_dtype(sg.Tensors(out)) == 3:
            ozp -= 128  # UINT8 -> INT8 domain shift, same as the weights
        out_quant = ", out_scale = %s : f32, out_zero_point = %d : i32" % (
            fnum(oscale), ozp)

        if kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            if kind == "CONV_2D":
                o = opts(op, schema.Conv2DOptions)
                pad = PADDING[o.Padding()]
                st = [int(o.StrideH()), int(o.StrideW())]
                fused = FUSED[int(o.FusedActivationFunction())]
            else:
                o = opts(op, schema.DepthwiseConv2DOptions)
                pad = PADDING[o.Padding()]
                st = [int(o.StrideH()), int(o.StrideW())]
                fused = FUSED[int(o.FusedActivationFunction())]
            wt = sg.Tensors(ins[1])
            wshape = [int(wt.Shape(i)) for i in range(wt.ShapeLength())]
            wvals, wscale, wzp = import_weights(model, ins[1], wt)
            bias, _ = read_buffer(model, ins[2])
            bscale = dequant_bias_scale(model, ins[0], ins[1])
            opname = "conv_2d" if kind == "CONV_2D" else "depthwise_conv_2d"
            extra = (
                ", weights = %s, bias = %s, strides = [%d, %d], padding = \"%s\", "
                "weight_scale = %s : f32, weight_zero_point = %d : i32, "
                "bias_scale = %s : f32"
                % (dense(wvals, wshape, "i8"), dense(bias, [len(bias)], "i32"),
                   st[0], st[1], pad, fnum(wscale), wzp, fnum(bscale))
            )
            b.add(nid, opname, in_names, extra)
        elif kind in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
            o = opts(op, schema.Pool2DOptions)
            pad = PADDING[o.Padding()]
            fused = FUSED[int(o.FusedActivationFunction())]
            opname = "average_pool_2d" if kind == "AVERAGE_POOL_2D" else "max_pool_2d"
            extra = (", window = [%d, %d], strides = [%d, %d], padding = \"%s\""
                     % (int(o.FilterHeight()), int(o.FilterWidth()),
                        int(o.StrideH()), int(o.StrideW()), pad))
            b.add(nid, opname, in_names, extra)
        elif kind == "RESHAPE":
            shape_data, shape_t = read_buffer(model, ins[1])
            new_shape = [int(v) for v in shape_data]
            extra = ", shape = [%s]" % ", ".join(str(v) for v in new_shape)
            b.add(nid, "reshape", in_names, extra)
        elif kind == "SOFTMAX":
            b.add(nid, "softmax", in_names)
        elif kind == "ADD":
            o = opts(op, schema.AddOptions)
            fused = FUSED[int(o.FusedActivationFunction())]
            b.add(nid, "add", in_names)
        elif kind == "FULLY_CONNECTED":
            o = opts(op, schema.FullyConnectedOptions)
            fused = FUSED[int(o.FusedActivationFunction())]
            wt = sg.Tensors(ins[1])
            wshape = [int(wt.Shape(i)) for i in range(wt.ShapeLength())]
            wvals, wscale, wzp = import_weights(model, ins[1], wt)
            bias, _ = read_buffer(model, ins[2])
            bscale = dequant_bias_scale(model, ins[0], ins[1])
            extra = (", weights = %s, bias = %s, weight_scale = %s : f32, "
                     "weight_zero_point = %d : i32, bias_scale = %s : f32"
                     % (dense(wvals, wshape, "i8"), dense(bias, [len(bias)], "i32"),
                        fnum(wscale), wzp, fnum(bscale)))
            b.add(nid, "fully_connected", in_names, extra)
        elif kind == "CONCATENATION":
            o = opts(op, schema.ConcatenationOptions)
            extra = ", axis = %d" % int(o.Axis())
            b.add(nid, "concatenation", in_names, extra)

        names[out] = nid
        # The source quantization of this operator's output tensor belongs on the
        # operator node itself (the generator keys output scales off the op that
        # produces the tensor), so attach it here for every node.
        b.nodes[-1] = b.nodes[-1][:-1] + out_quant + "}"
        if kind in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "ADD"):
            fused = infer_activation(sg, out, fused)
        if fused:
            aid = b.unique(nid + "_" + fused)
            b.add(aid, fused, [nid], out_quant=out_quant)
            names[out] = aid

    output_name = names.get(out_idx)
    if output_name is None:
        raise ImportError_("graph output tensor is not produced by any operator")

    # Materialise the MLIR module.
    params = "({}) -> ({})".format(
        ", ".join("d%d" % i for i in range(len(out_shape))),
        ", ".join("d%d" % i for i in range(len(out_shape))),
    )
    const_map = "affine_map<({}) -> ({})>".format(
        ", ".join("d%d" % i for i in range(len(out_shape))),
        ", ".join("0" for _ in in_shape),
    )
    id_map = "affine_map<({}) -> ({})>".format(
        ", ".join("d%d" % i for i in range(len(out_shape))),
        ", ".join("d%d" % i for i in range(len(out_shape))),
    )
    iters = ", ".join('"parallel"' for _ in out_shape)
    in_ty = tensor_ty(in_shape)
    out_ty = tensor_ty(out_shape)
    body = (
        f"    %init = arith.constant dense<0> : {out_ty}\n"
        f"    %r = linalg.generic {{indexing_maps = [{const_map}, {id_map}], "
        f"iterator_types = [{iters}]}} ins(%arg0 : {in_ty}) outs(%init : {out_ty}) {{\n"
        f"    ^bb0(%in: i8, %out: i8):\n"
        f"      linalg.yield %in : i8\n"
        f"    }} -> {out_ty}\n"
        f"    return %r : {out_ty}"
    )
    graph = '{output = "%s", nodes = [%s]}' % (output_name, ", ".join(b.nodes))
    interface = interface_quant_attrs(sg.Tensors(in_idx), sg.Tensors(out_idx))
    attrs = ('coral.operation = "graph", coral.tag = "%s", coral.graph = %s, %s'
             % (args.tag, graph, interface))
    mlir = (
        f"module @{args.func_name} {{\n"
        f"  func.func public @{args.func_name}(%arg0: {in_ty}) -> {out_ty}\n"
        f"      attributes {{{attrs}}} {{\n{body}\n  }}\n}}\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write(mlir)
    print("imported %s: %d ops, input %s, output %s -> %s (%d bytes)"
          % (args.model, len(b.nodes), in_shape, out_shape, args.output,
             len(mlir)))


if __name__ == "__main__":
    try:
        main()
    except ImportError_ as exc:
        print("IMPORT ERROR: %s" % exc, file=sys.stderr)
        sys.exit(2)
