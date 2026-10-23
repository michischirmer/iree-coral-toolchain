#!/usr/bin/env bash
# Example: import, compile and run MobileNetV2 on the Edge TPU.
#
#   examples/mobilenet_v2.sh [model.tflite]
#
# Defaults to the fetched test model.  Produces model.mlir and model.vmfb in
# the output directory and, when a device is present, runs one real image.
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

MODEL="${1:-$TOOLCHAIN_ROOT/tests/test_data/mobilenet_v2_1.0_224_quant.tflite}"
OUT="${OUT:-$WORKSPACE/coral-example}"
TAG="example"

require_file "$MODEL"
[[ -x "$IREE_COMPILE" ]] || die "iree-compile missing (run scripts/bootstrap_all.sh)"
[[ -x "$IREE_RUN_MODULE" ]] || die "iree-run-module missing (run scripts/bootstrap_all.sh)"

mkdir -p "$OUT"
MLIR="$OUT/model.mlir"
VMFB="$OUT/model.vmfb"

log "=== 1. import $MODEL ==="
"$PY" "$TOOLCHAIN_ROOT/tools/import_tflite.py" "$MODEL" -o "$MLIR" --tag "$TAG"

log ""
log "=== 2. compile -> $VMFB ==="
"$IREE_COMPILE" --iree-hal-target-backends=coral "$MLIR" -o "$VMFB"

log ""
log "=== 3. report ==="
log "  vmfb: $VMFB ($(stat -c%s "$VMFB") bytes)"
log "  generated model: $GENERATED/$TAG/model.tflite"
log "  Edge TPU model : $GENERATED/$TAG/model_edgetpu.tflite"

if [[ ! -e /dev/apex_0 ]]; then
  log ""
  log "no Edge TPU device; skipping inference"
  exit 0
fi

log ""
log "=== 4. run on the Edge TPU ==="
IMG="$TOOLCHAIN_ROOT/tests/test_data/cat.bmp"
if [[ -f "$IMG" ]]; then
  "$PY" - "$MODEL" "$IMG" "$OUT/input.bin" <<'EOF'
import sys
import numpy as np
import tensorflow as tf
model, img_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
det = tf.lite.Interpreter(model_path=model)
det.allocate_tensors()
idt = det.get_input_details()[0]
h, w = int(idt["shape"][1]), int(idt["shape"][2])
x = tf.image.decode_bmp(tf.io.read_file(img_path), channels=3)
x = tf.image.resize(x, [h, w])
real = (x.numpy().astype(np.float32) / 255.0) * 2.0 - 1.0
scale, zp = idt["quantization"]
info = np.iinfo(np.dtype(idt["dtype"]))
q = np.clip(np.round(real / scale) + zp, info.min, info.max).astype(idt["dtype"])
open(out, "wb").write(bytes((int(v) & 0xFF) for v in q.reshape(-1)))
print(f"wrote {out}: {h}x{w}x{idt['shape'][3]} input")
EOF
  # The runtime build stages libcoral_native.so and libedgetpu.so.1 beside
  # iree-run-module with an $ORIGIN RUNPATH, so no LD_LIBRARY_PATH is needed.
  "$IREE_RUN_MODULE" --device=local-sync --module="$VMFB" --function=run \
    "--input=1x224x224x3xi8=@$OUT/input.bin" | tail -3
else
  log "no test image; provide an input with:"
  log "  $IREE_RUN_MODULE --device=local-sync --module=$VMFB --function=run \\"
  log "      --input=1x224x224x3xi8=@input.bin"
fi
