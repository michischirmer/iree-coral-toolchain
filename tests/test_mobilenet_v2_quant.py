#!/usr/bin/env python3
"""Regression test: the generated MobileNetV2 must preserve the source model's
quantization *losslessly*, operator by operator.

This is the test that would have caught the import re-quantization regression
described in the project's MOBILENET_QUANTIZATION_FIX.md.  It compiles the
pretrained ``mobilenet_v2_1.0_224_quant.tflite`` through the native Coral
pipeline and then asserts, against the source model, that

  1. every convolution filter is byte-for-byte identical and dequantizes to the
     same real values (exact weights, exact per-tensor scale/zero point);
  2. every operator's output (activation) scale and zero point is identical;
  3. every operator's dequantized output on a real image is bit-exact
     (max absolute error 0, correlation 1.0).

Task 3 is checked with ``tools/compare_layers.py``, which reads each operator's
output from a model truncated after that operator.  Reading the output tensor of
a live interpreter is *not* reliable: TFLite reuses intermediate buffers, so by
the time ``invoke()`` returns the earlier operators' buffers hold later data.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import paths

IREE_COMPILE = paths.IREE_COMPILE
IMPORTER = paths.IMPORTER
COMPARE_LAYERS = paths.COMPARE_LAYERS
GENERATED = paths.GENERATED
SOURCE_MODEL = paths.SOURCE_MODEL
IMAGE = os.path.join(paths.TEST_DATA, "cat.bmp")
TAG = "mnet_v2_quant"


def log(msg):
    print(msg, flush=True)


def check(cond, msg):
    if not cond:
        raise SystemExit("FAIL: " + msg)
    log("  ok: " + msg)


def env_with_lib():
    return paths.compiler_env()


def main():
    import numpy as np
    import tensorflow as tf

    op_type = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    check(os.path.exists(SOURCE_MODEL), f"source model present: {SOURCE_MODEL}")

    workdir = tempfile.mkdtemp(prefix="coral_mnet2_quant_")
    mlir = os.path.join(workdir, "mobilenet_v2.mlir")
    vmfb = os.path.join(workdir, "mobilenet_v2.vmfb")

    log("\n=== import + compile ===")
    proc = subprocess.run([sys.executable, IMPORTER, SOURCE_MODEL, "-o", mlir,
                           "--tag", TAG], env=env_with_lib(),
                          capture_output=True, text=True)
    check(proc.returncode == 0,
          f"importer succeeded: {(proc.stdout + proc.stderr)[-500:]}")
    proc = subprocess.run([IREE_COMPILE, "--iree-hal-target-backends=coral",
                           mlir, "-o", vmfb], env=env_with_lib(),
                          capture_output=True, text=True)
    check(proc.returncode == 0,
          f"iree-compile succeeded: {(proc.stdout + proc.stderr)[-1000:]}")

    gen_model = os.path.join(GENERATED, TAG, "model.tflite")
    check(os.path.exists(gen_model), "generated TFLite model exists")

    orig = tf.lite.Interpreter(model_path=SOURCE_MODEL,
                               experimental_op_resolver_type=op_type)
    gen = tf.lite.Interpreter(model_path=gen_model,
                              experimental_op_resolver_type=op_type)
    orig.allocate_tensors()
    gen.allocate_tensors()
    oo, og = orig._get_ops_details(), gen._get_ops_details()

    log("\n=== 1. filters are byte-exact ===")
    n_filters = 0
    worst = 0.0
    for ao, ag in zip(oo, og):
        if ag["op_name"] not in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
            continue
        fo = orig.get_tensor(ao["inputs"][1])
        fg = gen.get_tensor(ag["inputs"][1])
        so, zo = orig.get_tensor_details()[ao["inputs"][1]]["quantization"]
        sg, zg = gen.get_tensor_details()[ag["inputs"][1]]["quantization"]
        if not (np.array_equal(fo, fg) and zo == zg and abs(so - sg) < 1e-12):
            raise SystemExit(
                f"FAIL: filter mismatch for op {ag['op_name']}: bytes_equal="
                f"{np.array_equal(fo, fg)} zp {zo}->{zg} scale {so}->{sg}")
        real_o = (fo.astype(np.float64) - zo) * so
        real_g = (fg.astype(np.float64) - zg) * sg
        worst = max(worst, float(np.abs(real_o - real_g).max()))
        n_filters += 1
    check(n_filters >= 50, f"checked {n_filters} convolution filters")
    check(worst == 0.0,
          f"every filter dequantizes identically (max real error {worst:.3e})")

    log("\n=== 2. activation scale / zero point identical ===")
    n_act = 0
    mismatch = []
    for k in range(min(len(oo), len(og))):
        to = orig.get_tensor_details()[oo[k]["outputs"][0]]
        tg = gen.get_tensor_details()[og[k]["outputs"][0]]
        so, zo = to["quantization"]
        sg, zg = tg["quantization"]
        if abs(so - sg) > 1e-12 or zo != zg:
            mismatch.append((k, og[k]["op_name"], so, zo, sg, zg))
        n_act += 1
    check(not mismatch,
          f"all {n_act} operator output quantizations match the source"
          + ("" if not mismatch else f" (mismatches: {mismatch[:5]})"))

    log("\n=== 3. per-operator dequantized outputs are bit-exact ===")
    proc = subprocess.run(
        [sys.executable, COMPARE_LAYERS, SOURCE_MODEL, gen_model, IMAGE,
         "--csv", os.path.join(workdir, "layers.csv")],
        env=env_with_lib(), capture_output=True, text=True)
    check(proc.returncode == 0,
          f"compare_layers succeeded: {(proc.stdout + proc.stderr)[-500:]}")
    rows = []
    for line in proc.stdout.splitlines():
        m = re.match(r"^(\d+)\s+(\S+)\s+([\dx]+)\s+.*?\s+([\d.]+)\s+([\d.]+)\s+"
                     r"([\d.]+)\s*$", line)
        if m:
            rows.append((int(m.group(1)), m.group(2), float(m.group(4)),
                         float(m.group(5)), float(m.group(6))))
    check(rows, "layer comparison produced per-operator results")
    max_err = max(r[2] for r in rows)
    min_corr = min(r[4] for r in rows)
    check(max_err == 0.0,
          f"every one of the {len(rows)} operators is bit-exact "
          f"(max abs error {max_err})")
    check(min_corr == 1.0,
          f"every operator correlates 1.0 with the source (min {min_corr:.6f})")

    log("\n=== 4. end-to-end classification agrees ===")
    real_img = tf.io.read_file(IMAGE)
    real_img = tf.image.decode_bmp(real_img, channels=3)
    real_img = tf.image.resize(real_img, [224, 224])
    real = (real_img.numpy().astype(np.float32) / 255.0) * 2.0 - 1.0
    real = real.reshape(1, 224, 224, 3)
    results = []
    for it in (orig, gen):
        det = it.get_input_details()[0]
        s, z = det["quantization"]
        info = np.iinfo(np.dtype(det["dtype"]))
        q = np.clip(np.round(real / s) + z, info.min, info.max).astype(
            det["dtype"])
        it.set_tensor(det["index"], q)
        it.invoke()
        odt = it.get_output_details()[0]
        sc, zz = odt["quantization"]
        results.append((it.get_tensor(odt["index"]).astype(np.float64) - zz) * sc)
    a, b = results
    check(int(np.argmax(a)) == int(np.argmax(b)),
          f"same top-1 class ({int(np.argmax(a))})")
    corr = float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])
    check(corr > 0.9999, f"logits correlate {corr:.6f}")

    log("\nQUANTIZATION REGRESSION OK: generated MobileNetV2 preserves the "
        "source model's quantization losslessly (exact filters, exact "
        "activation scales, bit-exact per-operator outputs).")
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
