#!/usr/bin/env python3
"""Run the generated MobileNetV2 on the physical Edge TPU, end to end.

Pipeline exercised:

  mobilenet_v2_1.0_224_quant.tflite          (pretrained, UINT8, 1001 classes)
    -> tools/import_tflite.py                (graph + quant metadata -> coral.graph)
    -> iree-compile --iree-hal-target-backends=coral
    -> ONE Edge TPU executable embedded in ONE relocatable VMFB
    -> native IREE Coral HAL on /dev/apex_0

Checks:
  * the real network maps to exactly one Edge TPU subgraph with every operator
    mapped (no CPU fallback);
  * the on-hardware result matches the TFLite-CPU result of the *same* generated
    UINT8 model within a small documented tolerance;
  * the generated model still behaves like the original pretrained MobileNet:
    the predicted class agrees on real inputs;
  * latency and artifact sizes are measured and reported.

Requires the physical device and the proprietary edgetpu_compiler; it exits
non-zero when either is absent so callers can skip it deliberately.
"""

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import paths

ROOT = paths.TOOLCHAIN_ROOT
IREE_COMPILE = paths.IREE_COMPILE
RUN_MODULE = paths.RUN_MODULE
EDGETPU_LIB = paths.EDGETPU_LIB
GENERATED = paths.GENERATED
APEX_DEVICE = paths.APEX_DEVICE
SOURCE_MODEL = paths.SOURCE_MODEL
IMPORTER = paths.IMPORTER
TEST_DATA = paths.TEST_DATA
TAG = "mnet_v2"
SHAPE = [1, 224, 224, 3]
IMAGES = ["cat.bmp", "grace_hopper.bmp", "bird.bmp", "owl.jpg", "parrot.jpg"]
# Edge TPU requantization rounding can move the top logits by a few LSBs; the
# measured maximum over the test images is 6 codes out of 255.
HW_TOLERANCE = 8


def log(msg):
    print(msg, flush=True)


def check(cond, msg):
    if not cond:
        raise SystemExit("FAIL: " + msg)
    log("  ok: " + msg)


def env_with_lib():
    # The generator/edgetpu roots the compiler needs are provided explicitly
    # (see paths.compiler_env). The runtime build stages libcoral_native.so and
    # libedgetpu.so.1 beside iree-run-module with an $ORIGIN RUNPATH, so nothing
    # needs to be added here. Only fall back to LD_LIBRARY_PATH when the staged
    # copies are absent (e.g. an out-of-tree runtime build), so the relocation
    # property is exercised by default.
    env = paths.compiler_env()
    staged = os.path.join(os.path.dirname(RUN_MODULE), "libedgetpu.so.1")
    if not os.path.exists(staged):
        env["LD_LIBRARY_PATH"] = EDGETPU_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


_OUTPUT_RE = re.compile(r"\b\d+(?:x\d+)*xi8=(.*)$")


def parse_output(combined):
    for line in combined.splitlines():
        m = _OUTPUT_RE.search(line.strip())
        if m:
            return [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    return None


def run_module(vmfb, shape, inputs, cwd=None):
    shape_s = "x".join(str(s) for s in shape)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        fh.write(bytes((v & 0xFF) for v in inputs))
        path = fh.name
    try:
        return subprocess.run(
            [RUN_MODULE, "--device=local-sync", f"--module={vmfb}",
             "--function=run", f"--input={shape_s}xi8=@{path}"],
            env=env_with_lib(), capture_output=True, text=True, cwd=cwd,
        )
    finally:
        os.unlink(path)


def load_image(path):
    import numpy as np
    import tensorflow as tf

    img = tf.io.read_file(path)
    if path.lower().endswith(".bmp"):
        img = tf.image.decode_bmp(img, channels=3)
    else:
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [SHAPE[1], SHAPE[2]])
    return (img.numpy().astype(np.float32) / 255.0) * 2.0 - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--latency-runs", type=int, default=20)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf

    check(os.path.exists(SOURCE_MODEL), f"source MobileNet present: {SOURCE_MODEL}")
    check(os.path.exists(APEX_DEVICE), "Edge TPU device /dev/apex_0 present")
    check(os.path.exists(IREE_COMPILE), f"iree-compile present: {IREE_COMPILE}")
    check(os.path.exists(RUN_MODULE), f"iree-run-module present: {RUN_MODULE}")

    workdir = tempfile.mkdtemp(prefix="coral_mnet2_c_")
    run_dir = tempfile.mkdtemp(prefix="coral_mnet2_r_")
    mlir = os.path.join(workdir, "mobilenet_v2.mlir")
    vmfb = os.path.join(workdir, "mobilenet_v2.vmfb")

    log("\n=== import: real quantized TFLite -> coral.graph ===")
    proc = subprocess.run(
        [sys.executable, IMPORTER, SOURCE_MODEL, "-o", mlir, "--tag", TAG],
        env=env_with_lib(), capture_output=True, text=True,
    )
    check(proc.returncode == 0,
          f"importer succeeded: {(proc.stdout + proc.stderr)[-800:]}")
    log("  " + proc.stdout.strip().splitlines()[-1])

    log("\n=== compile: coral.graph -> Edge TPU executable in a VMFB ===")
    t0 = time.time()
    proc = subprocess.run(
        [IREE_COMPILE, "--iree-hal-target-backends=coral", mlir, "-o", vmfb],
        env=env_with_lib(), capture_output=True, text=True,
    )
    compile_s = time.time() - t0
    check(proc.returncode == 0,
          f"iree-compile succeeded: {(proc.stdout + proc.stderr)[-1500:]}")

    log("\n=== mapping report ===")
    stdout_file = os.path.join(GENERATED, TAG, "model_edgetpu.stdout")
    log_file = os.path.join(GENERATED, TAG, "model_edgetpu.log")
    stdout_text = open(stdout_file).read()
    table = [ln for ln in open(log_file).read().splitlines()
             if "Mapped to Edge TPU" in ln or "Operation not supported" in ln]
    log("  " + "\n  ".join(table))
    check(table and all("Mapped to Edge TPU" in ln for ln in table),
          "every MobileNetV2 operator mapped to the Edge TPU (no CPU fallback)")
    m = re.search(r"Number of Edge TPU subgraphs:\s*(\d+)", stdout_text)
    check(m and int(m.group(1)) == 1,
          f"one Edge TPU subgraph ({m.group(0) if m else 'missing'})")
    m = re.search(r"Total number of operations:\s*(\d+)", stdout_text)
    check(m and int(m.group(1)) == 66,
          f"all {m.group(1) if m else '?'} operations in that subgraph")

    ref_model = os.path.join(GENERATED, TAG, "model.tflite")
    check(os.path.exists(ref_model), "generated TFLite model exists")
    interp = tf.lite.Interpreter(model_path=ref_model)
    interp.allocate_tensors()
    idt = interp.get_input_details()[0]
    odt = interp.get_output_details()[0]

    originals = [os.path.join(TEST_DATA, p) for p in IMAGES]
    originals = [p for p in originals if os.path.exists(p)]
    check(originals, "at least one real test image present")

    qinputs = []
    for path in originals:
        real = load_image(path).reshape(SHAPE)
        scale, zp = idt["quantization"]
        info = np.iinfo(np.dtype(idt["dtype"]))
        q = np.clip(np.round(real / scale) + zp, info.min, info.max).astype(
            idt["dtype"])
        qinputs.append(q.reshape(tuple(int(s) for s in idt["shape"])))

    orig = tf.lite.Interpreter(model_path=SOURCE_MODEL)
    orig.allocate_tensors()
    oidt = orig.get_input_details()[0]
    oodt = orig.get_output_details()[0]

    def orig_codes(x):
        scale, zp = oidt["quantization"]
        q = np.clip(np.round(x / scale) + zp, 0, 255).astype(np.uint8).reshape(
            tuple(int(s) for s in oidt["shape"]))
        orig.set_tensor(oidt["index"], q)
        orig.invoke()
        return orig.get_tensor(oodt["index"]).reshape(-1).astype(np.int64)

    refs, origs = [], []
    for i, q in enumerate(qinputs):
        interp.set_tensor(idt["index"], q)
        interp.invoke()
        refs.append(interp.get_tensor(odt["index"]).reshape(-1).astype(np.int64))
        origs.append(orig_codes(load_image(originals[i])))
    check(any(not np.array_equal(refs[0], r) for r in refs[1:]),
          "the test inputs produce different CPU outputs")

    dst = os.path.join(run_dir, "mobilenet_v2.vmfb")
    shutil.copy2(vmfb, dst)
    hidden = GENERATED + ".hidden"
    shutil.rmtree(hidden, ignore_errors=True)
    renamed = os.path.isdir(GENERATED)
    if renamed:
        os.rename(GENERATED, hidden)
    try:
        log("\n=== inference on the Edge TPU (native IREE Coral HAL) ===")
        for i, (q, ref, orig_code) in enumerate(zip(qinputs, refs, origs)):
            proc = run_module(dst, SHAPE, q.reshape(-1).tolist(), cwd=run_dir)
            combined = proc.stdout + proc.stderr
            check(proc.returncode == 0, f"input#{i}: iree-run-module exit 0")
            check("coral-loader: loaded Edge TPU executable" in combined,
                  f"input#{i}: Coral loader selected (no fallback)")
            got = parse_output(combined)
            check(got is not None, f"input#{i}: output parsed ({len(got)} classes)")
            codes = np.array(got, dtype=np.int64) & 0xFF
            maxdiff = int(np.abs(codes - ref).max())
            ndiff = int((codes != ref).sum())
            agree = int(np.argmax(codes)) == int(np.argmax(ref))
            check(maxdiff <= HW_TOLERANCE,
                  f"input#{i}: hardware output within {HW_TOLERANCE} codes of "
                  f"the generated-model CPU reference (max|diff|={maxdiff}, "
                  f"{ndiff}/1001 classes differ)")
            check(agree,
                  f"input#{i}: same predicted class as the generated model's "
                  f"CPU reference (class {int(np.argmax(codes))})")
            match = int(np.argmax(orig_code)) == int(np.argmax(codes))
            log(f"  input#{i} {os.path.basename(originals[i])}: original class "
                f"{int(np.argmax(orig_code))}, imported/hardware class "
                f"{int(np.argmax(codes))}, agreement={match}")
            check(match, f"input#{i}: hardware agrees with the ORIGINAL "
                         f"pretrained MobileNetV2")
    finally:
        if renamed:
            shutil.rmtree(GENERATED, ignore_errors=True)
            os.rename(hidden, GENERATED)

    log("\n=== latency / memory ===")
    lat = []
    for _ in range(args.latency_runs):
        t0 = time.time()
        run_module(dst, SHAPE, qinputs[0].reshape(-1).tolist(), cwd=run_dir)
        lat.append(time.time() - t0)
    tp = os.path.join(GENERATED, TAG, "model_edgetpu.tflite")
    sizes = {
        "vmfb_bytes": os.path.getsize(vmfb),
        "edge_tpu_model_bytes": os.path.getsize(tp),
        "generated_tflite_bytes": os.path.getsize(ref_model),
        "source_model_bytes": os.path.getsize(SOURCE_MODEL),
    }
    log(f"  end-to-end invoke latency (process launch + load + run), "
        f"n={len(lat)}: min {min(lat)*1e3:.1f} ms, median "
        f"{statistics.median(lat)*1e3:.1f} ms, max {max(lat)*1e3:.1f} ms")
    for k, v in sizes.items():
        log(f"  {k}: {v} bytes")
    log(f"  compile time (import+iree-compile): {compile_s:.1f} s")

    log("\nHARDWARE OK: real MobileNetV2-1.0-224 imported, compiled, and executed "
        "on the Edge TPU (one subgraph, all 66 operators mapped, agreeing with "
        "the original model's predictions).")
    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
