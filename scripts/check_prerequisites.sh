#!/usr/bin/env bash
# Check that a machine has everything the toolchain needs.
#
#   scripts/check_prerequisites.sh
#
# Reports OK / MISSING for each requirement and exits non-zero if a hard
# requirement is missing.  Optional items (the Edge TPU device, the gasket
# kernel module, the proprietary edgetpu_compiler) are reported as warnings so
# the same script is useful on a build-only machine.
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

FAIL=0
HARD=()
SOFT=()
PENDING=()

hard_ok()   { printf '  OK       %s\n' "$*"; }
hard_bad()  { printf '  MISSING  %s\n' "$*"; FAIL=1; HARD+=("$*"); }
soft_ok()   { printf '  OK       %s\n' "$*"; }
soft_bad()  { printf '  WARN     %s\n' "$*"; SOFT+=("$*"); }
# Items the bootstrap itself creates (the virtual environment, the fork, the
# fetched third-party inputs) are reported as pending rather than missing, so
# this script can run as step 1 of a fresh bootstrap without failing before
# those steps have had a chance to run.
pending()   { printf '  PENDING  %s\n' "$*"; PENDING+=("$*"); }

log "=== operating system ==="
if [[ -r /etc/os-release ]]; then
  # shellcheck source=/dev/null
  . /etc/os-release
  printf '  %s\n' "${PRETTY_NAME:-unknown}"
  case "${VERSION_ID:-}" in
    24.04) hard_ok "Ubuntu 24.04 (verified platform)" ;;
    "")    hard_bad "cannot determine distribution" ;;
    *)     soft_bad "Ubuntu ${VERSION_ID} (only 24.04 is verified)" ;;
  esac
else
  hard_bad "/etc/os-release not readable"
fi

log ""
log "=== architecture ==="
case "$(uname -m)" in
  x86_64|amd64) hard_ok "$(uname -m) (x86-64)" ;;
  *)            hard_bad "$(uname -m) (only x86-64 is supported)" ;;
esac

log ""
log "=== system build tools ==="
check_tool() { # name, version-cmd, minimum
  local name="$1" vcmd="$2" minimum="$3" have
  if ! command -v "$name" >/dev/null 2>&1; then
    hard_bad "$name (missing)"
    return
  fi
  have="$(eval "$vcmd" 2>/dev/null | head -1)"
  if [[ -n "$minimum" ]] && ! version_ge "$have" "$minimum"; then
    hard_bad "$name $have (need >= $minimum)"
  else
    hard_ok "$name $have"
  fi
}
check_tool cmake   'cmake --version | sed -n "s/^cmake version //p"'            "$CMAKE_MIN"
check_tool ninja   'ninja --version'                                            "$NINJA_MIN"
check_tool clang   'clang --version | sed -n "s/.*version \([0-9.]*\).*/\1/p"'   "$CLANG_MIN"
check_tool clang++ 'clang++ --version | sed -n "s/.*version \([0-9.]*\).*/\1/p"' "$CLANG_MIN"
check_tool gcc     'gcc -dumpfullversion'                                       "$GCC_MIN"
check_tool git     'git --version | sed -n "s/git version //p"'                 ""
check_tool python3 'python3 --version | sed -n "s/Python //p"'                  "$PYTHON_VERSION"
check_tool curl      'curl --version | head -1'                                  ""
check_tool tar       'tar --version | head -1'                                   ""
check_tool sha256sum 'sha256sum --version | head -1'                             ""

log ""
log "=== python virtual environment (created by scripts/setup_venv.sh) ==="
if [[ -x "$PY" ]]; then
  PYVER="$("$PY" --version 2>&1 | sed -n 's/Python //p')"
  if version_ge "$PYVER" "$PYTHON_VERSION"; then
    hard_ok "venv python $PYVER at $VENV"
  else
    hard_bad "venv python $PYVER (need >= $PYTHON_VERSION)"
  fi
  for mod in numpy tensorflow flatbuffers; do
    if "$PY" -c "import $mod" >/dev/null 2>&1; then
      VER="$("$PY" -c "import $mod,sys;print(getattr($mod,'__version__','?'))" 2>/dev/null)"
      hard_ok "python module $mod $VER"
    else
      hard_bad "python module $mod (run scripts/setup_venv.sh)"
    fi
  done
else
  pending "virtual environment at $VENV (run scripts/setup_venv.sh)"
fi

log ""
log "=== IREE fork (created by scripts/bootstrap_iree.sh) ==="
if [[ -d "$IREE_DIR/.git" ]]; then
  hard_ok "IREE fork at $IREE_DIR"
  if [[ -d "$IREE_DIR/compiler/plugins/target/Coral" ]]; then
    hard_ok "Coral compiler plugin present"
  else
    hard_bad "Coral compiler plugin missing (check out $IREE_CORAL_BRANCH)"
  fi
  if [[ -f "$IREE_DIR/runtime/src/iree/hal/local/loaders/coral_executable_loader.c" ]]; then
    hard_ok "Coral HAL executable loader present"
  else
    hard_bad "Coral HAL executable loader missing"
  fi
else
  pending "IREE fork at $IREE_DIR (run scripts/bootstrap_iree.sh)"
fi

log ""
log "=== proprietary Edge TPU components (fetched by scripts/fetch_third_party.sh) ==="
if [[ -x "$EDGETPU_SRC/compiler/x86_64/edgetpu_compiler" ]]; then
  V="$("$EDGETPU_SRC/compiler/x86_64/edgetpu_compiler" --version 2>&1 | head -1)"
  soft_ok "edgetpu_compiler $V"
else
  pending "edgetpu_compiler under $EDGETPU_SRC (fetched; needed for hardware tests)"
fi
if [[ -f "$LIBEDGETPU_DIR/libedgetpu.so.1" ]]; then
  soft_ok "libedgetpu.so.1 at $LIBEDGETPU_DIR"
else
  pending "libedgetpu.so.1 at $LIBEDGETPU_DIR (fetched)"
fi
if [[ -f "$LIBEDGETPU_HEADERS/edgetpu_c.h" ]]; then
  soft_ok "libedgetpu headers at $LIBEDGETPU_HEADERS"
else
  pending "libedgetpu headers at $LIBEDGETPU_HEADERS (fetched)"
fi

log ""
log "=== Edge TPU hardware (optional) ==="
if [[ -e /dev/apex_0 ]]; then
  soft_ok "/dev/apex_0 present"
else
  soft_bad "/dev/apex_0 absent (no physical Edge TPU; hardware tests will be skipped)"
fi
if [[ "$(lsmod 2>/dev/null | grep -c '^gasket')" -gt 0 ]]; then
  soft_ok "gasket kernel module loaded"
else
  soft_bad "gasket kernel module not loaded (required for the physical device)"
fi

log ""
if [[ "$FAIL" -eq 0 ]]; then
  log "PREREQUISITES OK: build requirements satisfied."
  if [[ ${#PENDING[@]} -gt 0 ]]; then
    log "  ${#PENDING[@]} item(s) pending; the bootstrap creates them next."
  fi
  if [[ ${#SOFT[@]} -gt 0 ]]; then
    log "  ${#SOFT[@]} optional item(s) missing; hardware tests may not run."
  fi
else
  log "PREREQUISITES FAILED: ${#HARD[@]} requirement(s) missing."
  log "Install the missing packages, then retry."
  exit 1
fi
