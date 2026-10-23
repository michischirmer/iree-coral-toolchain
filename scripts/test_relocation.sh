#!/usr/bin/env bash
# Relocation test: prove the toolchain builds and runs from a different
# directory than the one it was checked out in.
#
#   scripts/test_relocation.sh [--with-compiler] [--keep]
#
# The test never trusts a CMake build tree to be relocatable (it is not): it
# copies the *sources* to a fresh scratch directory and rebuilds there. All
# paths must therefore be resolved relative to the script/repository, never to
# the original checkout, and no absolute build path may leak into an artifact.
#
# Steps:
#   1. copy this repository (and, when present, the sibling IREE fork) to a
#      scratch root;
#   2. grep the copy for any reference to the original checkout path;
#   3. clean-rebuild the native runtime and the runtime in the copy;
#   4. check the recorded RUNPATH of every produced binary/library is relative
#      ($ORIGIN) and free of absolute paths;
#   5. run a MobileNetV2 VMFB (rebuilt if --with-compiler, else copied) with a
#      scrubbed environment (no LD_LIBRARY_PATH) from an unrelated cwd.
#
# --with-compiler also clean-rebuilds iree-compile in the copy, which is slow;
# without it the test reuses the compiler binary only to produce a VMFB and
# exercises relocation of the runtime, which is where absolute paths leaked.
set -uo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

WITH_COMPILER=0
KEEP=0
for arg in "$@"; do
  case "$arg" in
    --with-compiler) WITH_COMPILER=1 ;;
    --keep)          KEEP=1 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

SCRATCH="${IREE_CORAL_RELOCATION_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/iree-coral-reloc-XXXXXX")}"
FAIL=0
pass() { printf '  ok:   %s\n' "$*"; }
fail() { printf '  FAIL: %s\n' "$*" >&2; FAIL=1; }

cleanup() {
  if [[ "$KEEP" -eq 0 && "$FAIL" -eq 0 ]]; then
    rm -rf "$SCRATCH"
  else
    log "scratch kept at $SCRATCH"
  fi
}
trap cleanup EXIT

log "=== relocation test ==="
log "  source      : $TOOLCHAIN_ROOT"
log "  scratch     : $SCRATCH"

# --- 1. copy the sources ----------------------------------------------------
log ""
log "=== 1. copy sources to the scratch root ==="
mkdir -p "$SCRATCH"
COPY_TOOLCHAIN="$SCRATCH/$(basename "$TOOLCHAIN_ROOT")"
cp -a "$TOOLCHAIN_ROOT" "$COPY_TOOLCHAIN"
pass "copied toolchain -> $COPY_TOOLCHAIN"

COPY_FORK=""
if [[ -d "$IREE_DIR/compiler/plugins/target/Coral" ]]; then
  COPY_FORK="$SCRATCH/$(basename "$IREE_DIR")"
  if command -v rsync >/dev/null 2>&1; then
    # Copy only the working tree; build directories and the (multi-GB) git
    # history are never relocatable and are not needed to rebuild.
    rsync -a --exclude='/.git' "$IREE_DIR/" "$COPY_FORK/"
  else
    cp -a "$IREE_DIR" "$COPY_FORK"
    rm -rf "$COPY_FORK/.git"
  fi
  pass "copied IREE fork -> $COPY_FORK"
else
  warn "no IREE fork at $IREE_DIR; testing the toolchain only"
fi

# --- 2. the copy must not mention the original path -------------------------
log ""
log "=== 2. no reference to the original checkout ==="
HITS="$(grep -rIl --exclude-dir=.git -e "$TOOLCHAIN_ROOT" "$COPY_TOOLCHAIN" 2>/dev/null || true)"
if [[ -n "$HITS" ]]; then
  fail "the copied toolchain still references $TOOLCHAIN_ROOT:"
  printf '        %s\n' $HITS
else
  pass "no source file references the original toolchain path"
fi
if [[ -n "$COPY_FORK" ]]; then
  # The fork's own files may legitimately mention upstream docs paths; only
  # flag build/ script references to the old checkout.
  FHITS="$(grep -rIl --exclude-dir=.git -e "$IREE_DIR" "$COPY_FORK/build_tools" \
             "$COPY_FORK/compiler/plugins/target/Coral" 2>/dev/null || true)"
  if [[ -n "$FHITS" ]]; then
    fail "the copied fork still references $IREE_DIR:"
    printf '        %s\n' $FHITS
  else
    pass "no fork build/plugin file references the original checkout"
  fi
fi

# --- 3. clean rebuild in the copy -------------------------------------------
log ""
log "=== 3. clean rebuild in the scratch root ==="
export IREE_CORAL_WORKSPACE="$SCRATCH"
export IREE_CORAL_BUILD_DIR="$SCRATCH"
[[ -n "$COPY_FORK" ]] && export IREE_CORAL_SOURCE="$COPY_FORK"

# External inputs (TensorFlow source, edgetpu, third-party) are not part of the
# relocation subject; link them in when they are not already inside the copy.
for tree in tensorflow-source edgetpu third-party libedgetpu .venv; do
  if [[ -e "$WORKSPACE/$tree" && ! -e "$SCRATCH/$tree" ]]; then
    ln -s "$WORKSPACE/$tree" "$SCRATCH/$tree"
  fi
done
# The verified Eigen archive is a file, not a tree.
for archive in "$WORKSPACE"/eigen-*.tar.gz; do
  [[ -e "$archive" ]] || continue
  [[ -e "$SCRATCH/$(basename "$archive")" ]] || ln -s "$archive" "$SCRATCH/$(basename "$archive")"
done

if bash "$COPY_TOOLCHAIN/scripts/build_native_runtime.sh" >"$SCRATCH/native.log" 2>&1; then
  pass "native runtime rebuilt in the copy"
else
  fail "native runtime rebuild failed (see $SCRATCH/native.log)"
  tail -20 "$SCRATCH/native.log" | sed 's/^/        /'
fi

if bash "$COPY_TOOLCHAIN/scripts/build_runtime.sh" >"$SCRATCH/runtime.log" 2>&1; then
  pass "runtime rebuilt in the copy"
else
  fail "runtime rebuild failed (see $SCRATCH/runtime.log)"
  tail -20 "$SCRATCH/runtime.log" | sed 's/^/        /'
fi

if [[ "$WITH_COMPILER" -eq 1 ]]; then
  if bash "$COPY_TOOLCHAIN/scripts/build_compiler.sh" >"$SCRATCH/compiler.log" 2>&1; then
    pass "iree-compile rebuilt in the copy"
  else
    fail "iree-compile rebuild failed (see $SCRATCH/compiler.log)"
    tail -20 "$SCRATCH/compiler.log" | sed 's/^/        /'
  fi
fi

# --- 4. RUNPATHs must be relative -------------------------------------------
log ""
log "=== 4. recorded RUNPATHs are relative ==="
RUN_MODULE="$SCRATCH/$(basename "$RUNTIME_INSTALL_BUILD")/tools/iree-run-module"
NATIVE_SO="$COPY_TOOLCHAIN/runtime/libcoral_native.so"
if command -v readelf >/dev/null 2>&1; then
  # Accept only $ORIGIN-relative entries: an absolute entry would pin the
  # artifact to the build tree it happened to be produced in.
  check_rpath() {
    local bin="$1"
    local rpath entry absolute=0
    if [[ ! -e "$bin" ]]; then
      fail "missing artifact $bin"
      return
    fi
    rpath="$(readelf -d "$bin" 2>/dev/null | sed -n 's/.*(RUNPATH).*\[\(.*\)\]/\1/p')"
    if [[ -z "$rpath" ]]; then
      fail "$(basename "$bin"): no RUNPATH (would need LD_LIBRARY_PATH)"
      return
    fi
    local IFS=:
    for entry in $rpath; do
      # An empty entry means the current directory. CMake emits one for the
      # binaries that opt into IREE's SETUP_INSTALL_RPATH handling; the library
      # directory always precedes it, so only absolute entries are a defect.
      [[ -z "$entry" ]] && continue
      [[ "$entry" == '$ORIGIN'* ]] || absolute=1
    done
    if [[ "$absolute" -eq 1 ]]; then
      fail "$(basename "$bin"): RUNPATH contains an absolute path: $rpath"
    else
      pass "$(basename "$bin"): RUNPATH=$rpath"
    fi
  }
  check_rpath "$RUN_MODULE"
  check_rpath "$NATIVE_SO"
  # The compiler links the shared libIREECompiler.so and is therefore the
  # artifact most likely to carry a build-tree path.
  check_rpath "$IREE_COMPILE"
else
  warn "readelf not available; skipping RUNPATH check"
fi

# --- 5. run from an unrelated cwd with a scrubbed environment ---------------
log ""
log "=== 5. execute from an unrelated directory without LD_LIBRARY_PATH ==="
if [[ ! -x "$RUN_MODULE" ]]; then
  warn "runtime missing; skipping the execution check"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/iree-coral-reloc-run-XXXXXX")"
  # Produce a VMFB with whichever compiler is available: the one rebuilt in the
  # copy when --with-compiler was given, else the already-built one in the
  # original workspace.  Only if neither exists do we fall back to reusing a
  # previously generated artifact.
  VMFB=""
  MLIR="$WORK/model.mlir"
  RELOC_COMPILER="$SCRATCH/iree-compiler-build/tools/iree-compile"
  [[ -x "$RELOC_COMPILER" ]] || RELOC_COMPILER="$IREE_COMPILE"
  SRC_MODEL="$TOOLCHAIN_ROOT/tests/test_data/mobilenet_v2_1.0_224_quant.tflite"
  if [[ -x "$RELOC_COMPILER" && -f "$SRC_MODEL" && -x "$PY" ]]; then
    if "$PY" "$TOOLCHAIN_ROOT/tools/import_tflite.py" "$SRC_MODEL" \
         -o "$MLIR" --tag reloc_probe >/dev/null 2>&1 && \
       IREE_CORAL_TOOLS_ROOT="$TOOLCHAIN_ROOT/tools" \
       IREE_CORAL_PYTHON="$PY" \
       IREE_CORAL_GENERATED_ROOT="$WORK/generated" \
       IREE_CORAL_EDGETPU_COMPILER="$EDGETPU_COMPILER" \
         "$RELOC_COMPILER" --iree-hal-target-backends=coral \
           "$MLIR" -o "$WORK/reloc.vmfb" >/dev/null 2>&1; then
      VMFB="$WORK/reloc.vmfb"
    fi
  fi
  if [[ -z "$VMFB" ]]; then
    VMFB="$(find "$GENERATED" "$WORKSPACE/generated" -name '*.vmfb' 2>/dev/null | head -1)"
    [[ -n "$VMFB" ]] && cp "$VMFB" "$WORK/reloc.vmfb" && VMFB="$WORK/reloc.vmfb"
  fi

  if [[ -z "$VMFB" || ! -f "$VMFB" ]]; then
    warn "no VMFB available; skipping the hardware run"
  elif [[ ! -e /dev/apex_0 ]]; then
    warn "/dev/apex_0 absent; running without the Edge TPU"
    ( cd / && env -i PATH=/usr/bin:/bin HOME="$WORK" "$RUN_MODULE" --help ) >/dev/null 2>&1 \
      && pass "iree-run-module starts with a scrubbed environment" \
      || fail "iree-run-module failed to start with a scrubbed environment"
  else
    : >"$WORK/input.bin"
    head -c $((224 * 224 * 3)) /dev/zero >"$WORK/input.bin"
    OUT="$( cd / && env -i PATH=/usr/bin:/bin HOME="$WORK" "$RUN_MODULE" \
              --device=local-sync --module="$VMFB" --function=run \
              "--input=1x224x224x3xi8=@$WORK/input.bin" 2>&1 )"
    if grep -q "coral-loader: loaded Edge TPU executable" <<<"$OUT"; then
      pass "copied VMFB executed on the Edge TPU from an unrelated cwd"
    else
      fail "the relocated runtime did not reach the Edge TPU:"
      tail -5 <<<"$OUT" | sed 's/^/        /'
    fi
  fi
  [[ "$KEEP" -eq 0 && "$FAIL" -eq 0 ]] && rm -rf "$WORK"
fi

# --- summary ----------------------------------------------------------------
log ""
if [[ "$FAIL" -eq 0 ]]; then
  log "RELOCATION TEST PASSED"
else
  log "RELOCATION TEST FAILED"
  exit 1
fi
