#!/usr/bin/env bash
# Verify an installed toolchain end to end.
#
#   scripts/verify_installation.sh [--skip-hardware]
#
# Always verifies (no hardware required):
#   * iree-compile, iree-run-module and libcoral_native.so exist and run;
#   * the MobileNetV2 import quantisation is lossless: exact filters, exact
#     activation quantisation, bit-exact per-operator outputs (the regression
#     that the fix was written for).
#
# Additionally, when a physical Edge TPU is present (and not skipped):
#   * edgetpu_compiler maps all 66 operators to one subgraph;
#   * the hardware reproduces the original model's top-1 on the test images.
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

SKIP_HW=0
[[ "${1:-}" == "--skip-hardware" ]] && SKIP_HW=1

FAIL=0
pass() { printf '  ok: %s\n' "$*"; }
fail() { printf '  FAIL: %s\n' "$*" >&2; FAIL=1; }

log "=== 1. artifacts ==="
[[ -x "$IREE_COMPILE" ]] && pass "iree-compile: $IREE_COMPILE" || fail "iree-compile missing"
[[ -x "$IREE_RUN_MODULE" ]] && pass "iree-run-module: $IREE_RUN_MODULE" || fail "iree-run-module missing"
[[ -f "$CORAL_NATIVE_SO" ]] && pass "libcoral_native.so: $CORAL_NATIVE_SO" || fail "libcoral_native.so missing"
[[ -x "$PY" ]] && pass "venv python: $PY" || fail "venv python missing"

if [[ "$FAIL" -ne 0 ]]; then
  log ""
  log "VERIFICATION FAILED (missing build artifacts; run scripts/bootstrap_all.sh)"
  exit 1
fi

log ""
log "=== 2. lossless MobileNetV2 import (no hardware needed) ==="
MODEL="$TOOLCHAIN_ROOT/tests/test_data/mobilenet_v2_1.0_224_quant.tflite"
if [[ ! -f "$MODEL" ]]; then
  # The regression is the point of this verification, so fetch the test data
  # rather than silently passing without running it.
  log "  test model not present; fetching it"
  if ! "$TOOLCHAIN_ROOT/scripts/fetch_test_data.sh"; then
    fail "test model unavailable; cannot run the quantisation regression"
  fi
fi
if [[ -f "$MODEL" ]]; then
  if PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
       "$PY" "$TOOLCHAIN_ROOT/tests/test_mobilenet_v2_quant.py" 2>&1 | tail -25; then
    pass "MobileNetV2 quantisation regression passed"
  else
    fail "MobileNetV2 quantisation regression failed"
  fi
fi

log ""
log "=== 3. physical Edge TPU (optional) ==="
if [[ "$SKIP_HW" -eq 1 ]]; then
  log "  note: skipped (--skip-hardware)"
elif [[ ! -e /dev/apex_0 ]]; then
  log "  note: no /dev/apex_0; hardware verification skipped"
elif [[ ! -x "$EDGETPU_SRC/compiler/x86_64/edgetpu_compiler" ]]; then
  log "  note: edgetpu_compiler not available; hardware verification skipped"
elif PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
     "$PY" "$TOOLCHAIN_ROOT/tests/test_mobilenet_v2_hardware.py" 2>&1 | tail -30; then
  pass "MobileNetV2 Edge TPU hardware test passed"
else
  fail "MobileNetV2 Edge TPU hardware test failed"
fi

log ""
if [[ "$FAIL" -eq 0 ]]; then
  log "VERIFICATION OK"
else
  log "VERIFICATION FAILED"
  exit 1
fi
