"""Generate a fully-quantized INT8 TFLite model for a connected Coral graph.

The graph is a directed acyclic graph of *named* operations (``coral.graph`` in
the MLIR module). Every node consumes one or more named tensors and produces the
tensor named by its ``id``. The graph input tensor is ``"input"``; the graph
output tensor is named by ``spec["output"]``.

This script materialises the *already validated* graph as ONE TensorFlow
function, converts it to a single fully-int8 TFLite model and runs
``edgetpu_compiler`` on it. It runs at build time only - no Python is involved in
the inference path.

The operator set is intentionally the same one the C++ Coral pass validates: the
C++ side owns shape/connectivity validation and the Edge TPU mapping
verification, this script only materialises the already-validated graph.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Operators (each applies one graph node to its already-materialised inputs).
# ---------------------------------------------------------------------------

def _int8_to_real(weights, shape, node):
    """Fully dequantize a node's int8 weights to real values.

    ``real = (q - zero_point) * scale``.  The source model's per-tensor
    asymmetric UINT8 weights are imported losslessly as INT8 with a shifted zero
    point (``q = v - 128``, ``zero_point = z - 128``) and the original scale, so
    the reconstruction is exact (default ``zero_point = 0``, ``scale = 1.0``
    keeps the synthetic form exact).  A per-tensor integer weight zero point is
    supported; a per-channel list would be a future extension.
    """
    w = np.array(node["weights"], dtype=np.float32).reshape(shape)
    w_zp = float(node.get("weight_zero_point", 0))
    return (w - w_zp) * float(node.get("weight_scale", 1.0))


def _conv(x, node):
    out_c, k_h, k_w, in_c = [int(d) for d in node["filter_shape"]]
    # Spec/compiler use the TFLite OHWI layout; tf.nn.conv2d wants HWIO.
    # `weight_scale`/`bias_scale` dequantize an imported model whose integers
    # were stored with an asymmetric scale (default 1.0 = already real-valued).
    w_ohwi = _int8_to_real(node["weights"], (out_c, k_h, k_w, in_c), node)
    w_hwio = w_ohwi.transpose(1, 2, 3, 0)
    b = np.array(node["bias"], dtype=np.float32) * float(
        node.get("bias_scale", 1.0)
    )
    strides = [int(s) for s in node.get("strides", [1, 1])]
    y = tf.nn.conv2d(
        x, tf.constant(w_hwio),
        strides=[1, strides[0], strides[1], 1],
        padding=node.get("padding", "VALID"),
    )
    return y + b


def _depthwise_conv(x, node):
    # TFLite depthwise filter layout: [1, kH, kW, in_channels * multiplier].
    _, k_h, k_w, total = [int(d) for d in node["filter_shape"]]
    in_c = int(node["in_channels"])
    mult = int(node["depth_multiplier"])
    assert total == in_c * mult, (total, in_c, mult)
    # tf.nn.depthwise_conv2d wants [kH, kW, in_channels, multiplier].
    w = _int8_to_real(node["weights"], (k_h, k_w, in_c, mult), node)
    b = np.array(node["bias"], dtype=np.float32) * float(
        node.get("bias_scale", 1.0)
    )
    strides = [int(s) for s in node.get("strides", [1, 1])]
    y = tf.nn.depthwise_conv2d(
        x, tf.constant(w),
        strides=[1, strides[0], strides[1], 1],
        padding=node.get("padding", "VALID"),
    )
    return y + b


def _fc(x, node):
    in_f = int(node["in_features"])
    out_f = int(node["out_features"])
    w = _int8_to_real(node["weights"], (out_f, in_f), node)
    b = np.array(node["bias"], dtype=np.float32) * float(
        node.get("bias_scale", 1.0)
    )
    # FULLY_CONNECTED contracts the last dimension, so flatten everything except
    # the batch dimension first (materialised as a TFLite RESHAPE).
    flat = tf.reshape(x, [1, -1])
    return tf.matmul(flat, tf.constant(w), transpose_b=True) + tf.constant(b)


def _pool(x, node, kind):
    window = [int(w) for w in node.get("window", [2, 2])]
    strides = [int(s) for s in node.get("strides", window)]
    fn = tf.nn.max_pool2d if kind == "max" else tf.nn.avg_pool2d
    return fn(
        x,
        ksize=[1, window[0], window[1], 1],
        strides=[1, strides[0], strides[1], 1],
        padding=node.get("padding", "VALID"),
    )


def apply_node(node, inputs):
    op = node["op"]
    if op == "conv_2d":
        return _conv(inputs[0], node)
    if op == "depthwise_conv_2d":
        return _depthwise_conv(inputs[0], node)
    if op == "fully_connected":
        return _fc(inputs[0], node)
    if op == "relu":
        return tf.nn.relu(inputs[0])
    if op == "relu6":
        return tf.nn.relu6(inputs[0])
    if op == "logistic":
        return tf.nn.sigmoid(inputs[0])
    if op == "tanh":
        return tf.nn.tanh(inputs[0])
    if op == "softmax":
        return tf.nn.softmax(inputs[0], axis=-1)
    if op == "log_softmax":
        return tf.nn.log_softmax(inputs[0], axis=-1)
    if op == "max_pool_2d":
        return _pool(inputs[0], node, "max")
    if op == "average_pool_2d":
        return _pool(inputs[0], node, "avg")
    if op == "reshape":
        return tf.reshape(inputs[0], [int(d) for d in node["shape"]])
    if op == "add":
        return tf.add(inputs[0], inputs[1])
    if op == "mul":
        return tf.multiply(inputs[0], inputs[1])
    if op in ("concatenation", "concat"):
        return tf.concat(inputs, axis=int(node.get("axis", -1)))
    raise ValueError("unsupported layer op: %s" % op)


def _build(nodes, output):
    def f(x):
        values = {"input": x}
        for node in nodes:
            values[node["id"]] = apply_node(
                node, [values[name] for name in node["inputs"]]
            )
        return values[output]

    return f


def _filter_shape(node, op):
    """The TFLite filter shape a weight node must materialise."""
    if op == "fully_connected":
        return [int(node["out_features"]), int(node["in_features"])]
    return [int(d) for d in node["filter_shape"]]


def _relabel_int8_to_uint8(model):
    """Relabel every INT8 tensor as UINT8 (``q + 128``, ``zero_point + 128``).

    The shift is value preserving (``(q + 128) - (z + 128) == q - z``), so it
    changes nothing about the model's meaning.  It is needed before installing a
    source filter that carries a non-zero zero point: the TFLite INT8 kernels
    require a filter zero point of exactly 0, whereas the UINT8 kernels (used by
    the original asymmetric source models) accept an asymmetric filter.
    """
    for sg in model.subgraphs:
        for t in sg.tensors:
            if t.type != 9:  # INT8
                continue
            buf = model.buffers[t.buffer]
            if buf.data is not None and len(buf.data):
                data = np.frombuffer(bytes(buf.data), dtype=np.int8).astype(np.int16)
                buf.data = (data + 128).astype(np.uint8).tolist()
            if t.quantization is not None and t.quantization.zeroPoint is not None \
                    and len(t.quantization.zeroPoint):
                t.quantization.zeroPoint = [
                    int(z) + 128 for z in t.quantization.zeroPoint
                ]
            t.type = 3  # UINT8


def _install_exact_filters(model_bytes, weight_nodes, spec):
    """Overwrite the converted model's filters with the exact source values.

    The TensorFlow converter re-quantises every filter it is given (per channel,
    symmetrically), so a losslessly imported filter is still perturbed.  Here the
    converted model's convolution-like operators are matched, in order, to the
    spec's weight nodes and each filter tensor is replaced by the source values
    with the source scale and zero point.  The integer bias is re-derived because
    its unit is ``input_scale * weight_scale`` and the weight scale changes.

    Returns ``(model_bytes, num_installed)``; raises if the operator streams do
    not line up so a misalignment can never silently corrupt the weights.
    """
    import flatbuffers
    from tensorflow.lite.python import schema_py_generated as schema

    kind_by_code = {3: "CONV_2D", 4: "DEPTHWISE_CONV_2D", 8: "FULLY_CONNECTED"}
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(model_bytes, 0))
    sg = model.subgraphs[0]

    def builtin_code(op):
        code = model.operatorCodes[op.opcodeIndex]
        return code.builtinCode or code.deprecatedBuiltinCode

    generated = []
    for op in sg.operators:
        code = builtin_code(op)
        if code in kind_by_code:
            generated.append((kind_by_code[code], int(op.inputs[1]),
                              int(op.inputs[2]), int(op.inputs[0])))
    if len(generated) != len(weight_nodes):
        raise SystemExit(
            "exact filter install: converted model has %d convolution-like "
            "operators but the spec has %d weight nodes" % (len(generated),
                                                            len(weight_nodes))
        )

    # A non-zero source zero point cannot be stored in an INT8 filter, so the
    # whole model is relabelled to the UINT8 scheme the source used.
    to_uint8 = any(int(n.get("weight_zero_point", 0)) != 0 for n in weight_nodes)
    if to_uint8:
        _relabel_int8_to_uint8(model)

    # Pin the interface tensors to the *source* model's quantization. Otherwise
    # the converter invents a range for them (e.g. an input scale of 2/255 where
    # the source uses 1/128) and the very same byte stream means a different
    # real value, which perturbs every activation downstream.
    for det, t in zip((spec.get("input_quant"), spec.get("output_quant")),
                      (sg.tensors[int(sg.inputs[0])], sg.tensors[int(sg.outputs[0])])):
        if not det:
            continue
        z = int(det["zero_point"])
        t.quantization.scale = [float(det["scale"])]
        t.quantization.zeroPoint = [z + 128 if to_uint8 else z]

    for (kind, wi, bi, ii), node in zip(generated, weight_nodes):
        filter_t, bias_t = sg.tensors[wi], sg.tensors[bi]
        want = _filter_shape(node, node["op"])
        if list(filter_t.shape) != want:
            raise SystemExit(
                "exact filter install: filter shape mismatch for node '%s' "
                "(%s vs %s)" % (node["id"], list(filter_t.shape), want)
            )
        scale = float(node["weight_scale"])
        zero_point = int(node.get("weight_zero_point", 0))
        values = np.array(node["weights"], dtype=np.int16)

        # The spec carries the source model's exact INT32 bias in units of
        # ``source_input_scale * weight_scale``.  Now that the activation scales
        # have been pinned to the source, the effective input scale IS the source
        # input scale, so the source bias can be installed verbatim -- no
        # re-derivation from the converter's (differently scaled) bias is needed.
        in_scale = float(sg.tensors[ii].quantization.scale[0])
        new_bias = np.array(node["bias"], dtype=np.int32)

        if to_uint8:
            filter_t.type = 3  # UINT8
            model.buffers[filter_t.buffer].data = (
                (values + 128).astype(np.uint8).tolist())
            filter_t.quantization.zeroPoint = [zero_point + 128]
        else:
            filter_t.type = 9  # INT8
            model.buffers[filter_t.buffer].data = (
                values.astype(np.int8).view(np.uint8).tolist())
            filter_t.quantization.zeroPoint = [zero_point]
        filter_t.quantization.scale = [scale]
        filter_t.quantization.quantizedDimension = 0

        bias_t.type = 2  # INT32
        model.buffers[bias_t.buffer].data = new_bias.view(np.uint8).tolist()
        bias_t.quantization.scale = [in_scale * scale]
        bias_t.quantization.zeroPoint = [0]

    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output()), len(generated)


def _install_exact_activations(model_bytes, spec):
    """Pin every intermediate activation to the *source* model's quantization.

    The converter derives activation ranges by calibrating the real-valued
    graph, so each intermediate tensor gets a range of its own choosing.  Those
    ranges differ from the source model's, and because a requantization step is
    ``round(acc * in_scale * weight_scale / out_scale)`` a differing output
    scale visibly changes the codes -- it is the dominant residual error once the
    weights are exact.

    Matching the source's scales removes it.  Operators are matched to spec
    nodes by *operator type and order within that type*, which lines up exactly
    for the imported graphs; a shape assertion guards against misalignment.

    Returns the new model bytes.
    """
    import flatbuffers
    from tensorflow.lite.python import schema_py_generated as schema

    name_by_kind = {
        "CONV_2D": "conv_2d",
        "DEPTHWISE_CONV_2D": "depthwise_conv_2d",
        "FULLY_CONNECTED": "fully_connected",
        "ADD": "add",
        "MUL": "mul",
        "AVERAGE_POOL_2D": "average_pool_2d",
        "MAX_POOL_2D": "max_pool_2d",
        "RESHAPE": "reshape",
        "SOFTMAX": "softmax",
    }
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(model_bytes, 0))
    sg = model.subgraphs[0]

    def builtin_code(op):
        code = model.operatorCodes[op.opcodeIndex]
        return code.builtinCode or code.deprecatedBuiltinCode

    # spec nodes carrying a source output quantization, grouped by op name
    wanted = {}
    for node in spec["nodes"]:
        if "out_scale" in node:
            wanted.setdefault(node["op"], []).append(node)

    # generated operators grouped by their spec op name, in graph order
    seen = {}
    installed = 0
    for op in sg.operators:
        kind = name_by_kind.get(_tflite_op_name(builtin_code(op)))
        if kind is None or kind not in wanted:
            continue
        nodes = wanted[kind]
        rank = seen.get(kind, 0)
        if rank >= len(nodes):
            continue
        seen[kind] = rank + 1
        node = nodes[rank]
        out_t = sg.tensors[int(op.outputs[0])]
        want_shape = [int(v) for v in node["out_shape"]]
        got_shape = [int(v) for v in out_t.shape] if out_t.shape is not None else []
        if got_shape != want_shape:
            raise SystemExit(
                "exact activation install: tensor shape mismatch for node "
                "'%s' (%s vs %s)" % (node["id"], got_shape, want_shape)
            )
        scale = float(node["out_scale"])
        # out_zero_point travels in the INT8 domain (as emitted by the pass).
        # _install_exact_filters relabels the whole model to UINT8 afterwards,
        # which shifts every zero point by +128 -- exactly the byte-transparent
        # transform the source interface needs.
        zp = int(node["out_zero_point"])
        if out_t.quantization is None:
            out_t.quantization = schema.QuantizationParametersT()
        out_t.quantization.scale = [scale]
        out_t.quantization.zeroPoint = [zp]
        out_t.quantization.quantizedDimension = 0
        installed += 1

    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output()), installed


def _tflite_op_name(code):
    return {
        3: "CONV_2D",
        4: "DEPTHWISE_CONV_2D",
        8: "FULLY_CONNECTED",
        0: "ADD",
        2: "MUL",
        1: "AVERAGE_POOL_2D",
        17: "MAX_POOL_2D",
        22: "RESHAPE",
        25: "SOFTMAX",
    }.get(code, "")


def _validate_model(model_bytes):
    """Fail loudly if the generated model cannot even be prepared by TFLite."""
    interp = tf.lite.Interpreter(
        model_content=model_bytes,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES),
    )
    interp.allocate_tensors()


def generate(spec):
    input_shape = [int(d) for d in spec["input_shape"]]
    nodes = spec["nodes"]
    output = spec["output"]
    assert nodes, "graph must contain at least one node"
    weight_nodes = [n for n in nodes if "weights" in n]
    # A spec that carries source zero points is a faithful import; its exact
    # filters are installed after conversion so no re-quantisation error remains.
    exact_import = bool(spec.get("exact_weights",
                                 any("weight_zero_point" in n for n in weight_nodes)))

    concrete = tf.function(_build(nodes, output), autograph=False).get_concrete_function(
        tf.TensorSpec(input_shape, tf.float32)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    def representative_dataset():
        """Calibration data for post-training quantization.

        The synthetic edge cases pin the extremal ranges; the bulk of the samples
        are low-frequency, natural-image-like inputs (smooth low-resolution noise
        upsampled plus a little high-frequency noise).  Random per-pixel noise
        alone over-estimates the activation ranges of a deep convolutional net
        (every intermediate tensor is driven by an implausible input), which
        coarsens the intermediate quantization and lets the quantized network
        drift far from the real-valued one.
        """
        rng = np.random.default_rng(1234)
        yield [np.zeros(input_shape, dtype=np.float32)]
        yield [np.full(input_shape, 1.0, dtype=np.float32)]
        yield [np.full(input_shape, -1.0, dtype=np.float32)]
        base = np.linspace(
            -1.0, 1.0, num=int(np.prod(input_shape)), dtype=np.float32
        ).reshape(input_shape)
        yield [base]
        rank4 = len(input_shape) == 4
        for _ in range(24):
            if rank4:
                h, w, c = input_shape[1], input_shape[2], input_shape[3]
                coarse = rng.uniform(-1.0, 1.0, size=(1, 8, 8, c)).astype(np.float32)
                img = tf.image.resize(coarse, [h, w], method="bicubic").numpy()
                img = img + rng.normal(0.0, 0.05, size=img.shape).astype(np.float32)
                yield [np.clip(img, -1.0, 1.0).astype(np.float32)]
            else:
                yield [rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model = converter.convert()

    interp = tf.lite.Interpreter(model_content=model)
    interp.allocate_tensors()
    ops = [op["op_name"] for op in interp._get_ops_details()]
    if not any(op in ops for op in
               ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED")):
        raise SystemExit("generated graph model has no recognisable operators: %s" % ops)
    print("generated graph ops: %s" % ops)

    if exact_import:
        model, act = _install_exact_activations(model, spec)
        model, installed = _install_exact_filters(model, weight_nodes, spec)
        _validate_model(model)
        print("installed exact source filters for %d operators, exact source "
              "activation quantization for %d operators" % (installed, act))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="JSON specification path")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    spec = json.load(open(args.spec))
    model = generate(spec)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as fh:
        fh.write(model)
    print("generated %s: %d bytes" % (args.output, len(model)))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    # The proprietary Edge TPU compiler. Overridable so the toolchain can run
    # the generator from any checkout; the default sits beside the source tree.
    edgetpu_compiler = os.environ.get("IREE_CORAL_EDGETPU_COMPILER")
    if not edgetpu_compiler:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        edgetpu_compiler = os.path.join(
            here, "edgetpu", "compiler", "x86_64", "edgetpu_compiler")
    cmd = [
        edgetpu_compiler,
        "-s",
        args.output,
        "-o",
        out_dir,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    # NOTE: the authoritative Edge TPU mapping verification happens in the C++
    # pass (it rejects partial mappings with a precise diagnostic). This script
    # only reports the compiler output and fails if the compiler itself failed.
    log = os.path.join(
        out_dir, os.path.basename(args.output).replace(".tflite", "_edgetpu.log")
    )
    if os.path.exists(log):
        print(open(log).read())


if __name__ == "__main__":
    sys.exit(main() or 0)
