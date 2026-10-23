#!/usr/bin/env bash
# One-shot bootstrap: reproduce the complete working toolchain on a fresh
# Ubuntu 24.04 machine.
#
#   scripts/bootstrap_all.sh [--repo URL]
#
# Runs, in order:
#   1. prerequisites check
#   2. python virtual environment
#   3. external inputs (TensorFlow source, Coral edgetpu, libedgetpu, Eigen)
#   4. IREE fork checkout at the pinned commit
#   5. the Coral-enabled compiler
#   6. libcoral_native.so (TFLite shim)
#   7. the Coral-enabled IREE runtime (iree-run-module)
#   8. test data (model + images)
#   9. verification (the MobileNetV2 quantisation regression, always; the
#      physical Edge TPU test, when a device is present)
#
# Every step is incremental and safe to re-run.  The fork is cloned from the
# URL pinned in versions.lock (IREE_CORAL_REPO); use --repo to override it.
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

REPO=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

log "############################################################"
log "# IREE Coral Edge TPU toolchain bootstrap"
log "#   workspace : $WORKSPACE"
log "#   toolchain : $TOOLCHAIN_ROOT"
log "#   IREE fork : $IREE_DIR"
log "############################################################"

log ""
log "########## step 1/9: prerequisites ##########"
"$TOOLCHAIN_ROOT/scripts/check_prerequisites.sh"

log ""
log "########## step 2/9: python environment ##########"
"$TOOLCHAIN_ROOT/scripts/setup_venv.sh"

log ""
log "########## step 3/9: external inputs ##########"
"$TOOLCHAIN_ROOT/scripts/fetch_third_party.sh"

log ""
log "########## step 4/9: IREE fork ##########"
if [[ -n "$REPO" ]]; then
  "$TOOLCHAIN_ROOT/scripts/bootstrap_iree.sh" --repo "$REPO"
else
  "$TOOLCHAIN_ROOT/scripts/bootstrap_iree.sh"
fi

log ""
log "########## step 5/9: compiler ##########"
"$TOOLCHAIN_ROOT/scripts/build_compiler.sh"

log ""
log "########## step 6/9: native runtime ##########"
"$TOOLCHAIN_ROOT/scripts/build_native_runtime.sh"

log ""
log "########## step 7/9: IREE runtime ##########"
"$TOOLCHAIN_ROOT/scripts/build_runtime.sh"

log ""
log "########## step 8/9: test data ##########"
"$TOOLCHAIN_ROOT/scripts/fetch_test_data.sh"

log ""
log "########## step 9/9: verification ##########"
"$TOOLCHAIN_ROOT/scripts/verify_installation.sh"

log ""
log "############################################################"
log "# BOOTSTRAP COMPLETE"
log "############################################################"
