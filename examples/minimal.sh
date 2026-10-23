#!/usr/bin/env bash
# Minimal end-to-end example: import, compile and run a quantized model on a
# Coral Edge TPU.
#
#   examples/minimal.sh [model.tflite] [input.bin]
#
# With no arguments it uses the fetched reference MobileNetV2 model and a zero
# input tensor, so the complete pipeline runs without any preprocessing:
#
#   tools/import_tflite.py -> iree-compile (coral) -> iree-run-module
#
# Pass a raw input tensor file to run real data (examples/mobilenet_v2.sh does
# the image preprocessing for the reference model).  model.mlir and model.vmfb
# are written under $OUT (default: <workspace>/coral-example).
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

MODEL="${1:-$TOOLCHAIN_ROOT/tests/test_data/mobilenet_v2_1.0_224_quant.tflite}"
INPUT="${2:-}"
OUT="${OUT:-$WORKSPACE/coral-example}"
# The reference model's input tensor; override for a different model.
INPUT_SPEC="${INPUT_SPEC:-1x224x224x3xi8}"

require_file "$MODEL"
[[ -x "$IREE_COMPILE" ]] || die "iree-compile missing (run scripts/bootstrap_all.sh)"
[[ -x "$IREE_RUN_MODULE" ]] || die "iree-run-module missing (run scripts/bootstrap_all.sh)"

mkdir -p "$OUT"
MLIR="$OUT/model.mlir"
VMFB="$OUT/model.vmfb"

log "=== 1. import $MODEL ==="
"$PY" "$TOOLCHAIN_ROOT/tools/import_tflite.py" "$MODEL" -o "$MLIR" --tag minimal

log ""
log "=== 2. compile -> $VMFB ==="
"$IREE_COMPILE" --iree-hal-target-backends=coral "$MLIR" -o "$VMFB"
log "  $VMFB ($(stat -c%s "$VMFB") bytes)"

if [[ ! -e /dev/apex_0 ]]; then
  log ""
  log "no Edge TPU device; skipping inference"
  exit 0
fi

if [[ -z "$INPUT" ]]; then
  INPUT="$OUT/input.bin"
  truncate -s 150528 "$INPUT"   # 1x224x224x3 int8, all zeros
  log ""
  log "no input given; using a zero tensor ($INPUT)"
fi
require_file "$INPUT"

log ""
log "=== 3. run on the Edge TPU ==="
"$IREE_RUN_MODULE" --device=local-sync --module="$VMFB" --function=run \
  "--input=$INPUT_SPEC=@$INPUT"
