#!/usr/bin/env python3
"""Layer-by-layer dequantized comparison of two fully-quantized TFLite models.

For two models that are meant to compute the same function, this walks both
operator streams in order and, for every *corresponding* operator, records:

  * operator type,
  * input/output shapes,
  * input/output (scale, zero point) for both models,
  * the maximum and mean absolute error between the two operators' dequantized
    outputs,
  * the correlation of the two dequantized outputs.

Values are compared in **real space** (dequantized), and both models are driven
by the same real-valued image, each quantized with *its own* input scheme, so a
differing input quantization shows up as a numerical difference rather than as a
spurious scale mismatch.

Buffer-reuse caveat
-------------------
TFLite reuses intermediate buffers, so reading ``get_tensor(op_output)`` after a
full ``invoke()`` can return a *later* operator's data (the buffer has already
been overwritten).  To observe an operator's true output this tool re-builds a
model truncated after that operator and runs it; the last operator's output is
then guaranteed to be the tensor of interest.  This is the only reliable way to
compare per-operator outputs.

The operators are aligned by position in the stream (the two models are the
same network), and the tool refuses to compare operators whose output shapes
disagree.

Usage:
    compare_layers.py ORIG.tflite GEN.tflite IMAGE [--csv out.csv] [--limit N]
"""

import argparse
import sys

import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

O = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES


def load_image(path, h, w):
    img = tf.io.read_file(path)
    if path.lower().endswith(".bmp"):
        img = tf.image.decode_bmp(img, channels=3)
    else:
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [h, w])
    return (img.numpy().astype(np.float32) / 255.0) * 2.0 - 1.0


def quantize(real, det):
    info = np.iinfo(np.dtype(det["dtype"]))
    scale, zp = det["quantization"]
    q = np.round(real / scale) + zp
    return np.clip(q, info.min, info.max).astype(det["dtype"])


def parse_model(path):
    raw = open(path, "rb").read()
    return schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(raw, 0))


def truncated_interpreter(modelT, nops):
    """Interpreter running only the first ``nops`` operators of subgraph 0."""
    sg0 = modelT.subgraphs[0]
    ops = sg0.operators[:nops]
    out_idx = int(ops[-1].outputs[0])

    m = schema.ModelT()
    m.version = modelT.version
    m.description = modelT.description
    m.operatorCodes = modelT.operatorCodes
    m.buffers = modelT.buffers
    m.metadata = modelT.metadata
    sg = schema.SubGraphT()
    sg.tensors = sg0.tensors
    sg.operators = ops
    sg.inputs = [int(sg0.inputs[0])]
    sg.outputs = [out_idx]
    sg.name = "truncated"
    m.subgraphs = [sg]
    m.signatureDefs = []

    b = flatbuffers.Builder(0)
    b.Finish(m.Pack(b), file_identifier=b"TFL3")
    return tf.lite.Interpreter(model_content=bytes(b.Output()),
                               experimental_op_resolver_type=O)


def snapshot(modelT, path, feed, nops):
    """Run the truncated model on ``feed`` and return (dequantized_out, qparam)."""
    it = truncated_interpreter(modelT, nops)
    it.allocate_tensors()
    det = it.get_input_details()[0]
    it.set_tensor(int(det["index"]), feed)
    it.invoke()
    od = it.get_output_details()[0]
    detail = it.get_tensor_details()[int(od["index"])]
    scale, zp = detail["quantization"]
    raw = it.get_tensor(int(od["index"])).astype(np.float64)
    return (raw - zp) * scale, (float(scale), int(zp))


def compare(orig_path, gen_path, image, limit):
    mo = parse_model(orig_path)
    mg = parse_model(gen_path)
    io = tf.lite.Interpreter(model_path=orig_path, experimental_op_resolver_type=O)
    io.allocate_tensors()
    det = io.get_input_details()[0]
    real = load_image(image, int(det["shape"][1]), int(det["shape"][2])).reshape(
        tuple(int(x) for x in det["shape"]))
    feed = quantize(real, det)

    ops_o = io._get_ops_details()
    n = min(len(ops_o), len(mg.subgraphs[0].operators))
    if limit:
        n = min(n, limit)

    rows = []
    for k in range(n):
        ao = ops_o[k]
        oi = int(ao["outputs"][0])
        gi = int(mg.subgraphs[0].operators[k].outputs[0])
        to = io.get_tensor_details()[oi]
        tg = [t for t in _tensor_details(mg) if t["index"] == gi]
        so = tuple(int(x) for x in to["shape"])
        sg = tuple(int(x) for x in tg[0]["shape"]) if tg else None
        if sg is None or so != sg:
            rows.append((k, ao["op_name"], None, so, sg, None, None, None, None,
                         None, None, None))
            continue
        vo, qo = snapshot(mo, orig_path, feed, k + 1)
        vg, qg = snapshot(mg, gen_path, feed, k + 1)
        diff = np.abs(vo - vg)
        denom = float(np.abs(vo).max()) + 1e-12
        if vo.size > 1 and vo.std() > 0 and vg.std() > 0:
            corr = float(np.corrcoef(vo.reshape(-1), vg.reshape(-1))[0, 1])
        else:
            corr = float("nan")
        rows.append((k, ao["op_name"], ao["op_name"], so, sg, None, None,
                     qo, qg, float(diff.max()), float(diff.mean()), corr,
                     denom))
    return rows


def _tensor_details(modelT):
    out = []
    for i, t in enumerate(modelT.subgraphs[0].tensors):
        shape = t.shape if t.shape is not None else []
        q = t.quantization
        scale = float(q.scale[0]) if q is not None and q.scale is not None \
            and len(q.scale) > 0 else 0.0
        zp = int(q.zeroPoint[0]) if q is not None and q.zeroPoint is not None \
            and len(q.zeroPoint) > 0 else 0
        out.append({"index": i, "shape": shape, "q": (scale, zp)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig")
    ap.add_argument("gen")
    ap.add_argument("image")
    ap.add_argument("--csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = compare(args.orig, args.gen, args.image, args.limit)
    print("%-3s %-18s %-14s %-26s %-26s %-9s %-9s %-8s" %
          ("op", "type", "out shape", "out (o|g) scale/zp",
           "orig out scale/zp", "maxabs", "meanabs", "corr"))
    first_bad = None
    for r in rows:
        (k, on, gn, so, sg, _, _, qo, qg, mx, mn, corr, denom) = r
        if qo is None:
            print("%-3d %-18s SHAPE MISMATCH %s vs %s" % (k, on, so, sg))
            continue
        flag = ""
        if denom > 0 and mx / denom > 0.02 and np.isfinite(corr) and corr < 0.99:
            flag = "  <-- divergence"
            if first_bad is None and (len(so) < 4 or so[1] > 7):
                first_bad = (k, on, so, mx, corr)
        print("%-3d %-18s %-14s %-26s %-26s %-9.4f %-9.5f %-8.4f%s"
              % (k, on, "x".join(map(str, sg)),
                 "%.6f/%d" % (qo[0], qo[1]), "%.6f/%d" % (qg[0], qg[1]),
                 mx, mn, corr, flag))
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["op", "type", "out_shape", "orig_scale", "orig_zp",
                        "gen_scale", "gen_zp", "max_abs_err", "mean_abs_err",
                        "corr"])
            for r in rows:
                if r[7] is None:
                    continue
                (k, on, gn, so, sg, _, _, qo, qg, mx, mn, corr, denom) = r
                w.writerow([k, on, "x".join(map(str, so)), qo[0], qo[1],
                            qg[0], qg[1], mx, mn, corr])
    if first_bad:
        print("\nfirst significant divergence: op %d (%s) shape %s "
              "maxabs=%.4f corr=%.4f" % first_bad)
    else:
        print("\nno operator crossed the divergence threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
