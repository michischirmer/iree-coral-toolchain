#!/usr/bin/env bash
# Build and install the Coral-enabled IREE runtime, with iree-run-module.
#
#   scripts/build_runtime.sh [--fresh]
#
# The runtime is configured with -DIREE_HAL_EXECUTABLE_LOADER_CORAL=ON, which
# compiles the fork's coral_executable_loader.c and links it against
# libcoral_native.so and libedgetpu.so.1.  iree-run-module is what actually
# executes the generated VMFB on the Edge TPU.
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

FRESH=0
[[ "${1:-}" == "--fresh" ]] && FRESH=1

require_cmd cmake ninja clang clang++ git
[[ -d "$IREE_DIR" ]] || die "IREE fork not found at $IREE_DIR (run scripts/bootstrap_iree.sh)"
[[ -f "$CORAL_NATIVE_SO" ]] || \
  die "libcoral_native.so not found (run scripts/build_native_runtime.sh first)"
[[ -f "$LIBEDGETPU_DIR/libedgetpu.so.1" ]] || \
  die "libedgetpu.so.1 not found at $LIBEDGETPU_DIR"

if [[ "$FRESH" -eq 1 ]]; then
  log "=== --fresh: removing $RUNTIME_INSTALL_BUILD ==="
  rm -rf "$RUNTIME_INSTALL_BUILD" "$RUNTIME_INSTALL"
fi

log "=== configure ($RUNTIME_INSTALL_BUILD) ==="
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" NINJAFLAGS="-j$JOBS" MAX_JOBS="$JOBS"
cmake -S "$IREE_DIR" -B "$RUNTIME_INSTALL_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX="$RUNTIME_INSTALL" \
  -DCMAKE_BUILD_RPATH="\$ORIGIN" \
  -DCMAKE_INSTALL_RPATH="\$ORIGIN" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_BENCHMARKS=OFF \
  -DIREE_BUILD_PYTHON_BINDINGS=OFF \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_HAL_EXECUTABLE_LOADER_CORAL=ON \
  -DIREE_HAL_EXECUTABLE_LOADER_CORAL_NATIVE_LIBRARY="$CORAL_NATIVE_SO" \
  -DIREE_HAL_EXECUTABLE_LOADER_CORAL_NATIVE_INCLUDE_DIR="$TOOLCHAIN_ROOT/runtime" \
  -DIREE_HAL_EXECUTABLE_LOADER_CORAL_EDGETPU_LIBRARY="$LIBEDGETPU_DIR/libedgetpu.so.1"

log ""
log "=== build iree-run-module (-j$JOBS) ==="
cmake --build "$RUNTIME_INSTALL_BUILD" --target iree-run-module --parallel "$JOBS"

require_file "$IREE_RUN_MODULE"
log ""
log "built $IREE_RUN_MODULE"

# Make the binary self-contained: place the native Coral runtime and libedgetpu
# beside it so the $ORIGIN RUNPATH finds them without LD_LIBRARY_PATH, and so
# the installed tree can be relocated as a whole.
BIN_DIR="$(dirname "$IREE_RUN_MODULE")"
log ""
log "=== stage the native runtime beside the binary ==="
cp -f "$CORAL_NATIVE_SO" "$BIN_DIR/libcoral_native.so"
# libcoral_native.so resolves libedgetpu through its own $ORIGIN RUNPATH, so
# both libraries must live in the same directory.
cp -f "$LIBEDGETPU_DIR/libedgetpu.so.1" "$BIN_DIR/libedgetpu.so.1"
ln -sf libedgetpu.so.1 "$BIN_DIR/libedgetpu.so"
log "  staged libcoral_native.so, libedgetpu.so.1 -> $BIN_DIR"

# Confirm the Coral loader really is linked in; this is the whole point.
if "$IREE_RUN_MODULE" --help 2>&1 | grep -qi coral; then
  log "  OK       iree-run-module references the Coral loader"
elif strings "$IREE_RUN_MODULE" 2>/dev/null | grep -q "coral-loader"; then
  log "  OK       iree-run-module contains the coral-loader marker"
else
  log "  note     could not confirm the Coral loader by name (still linked statically)"
fi

# Report the recorded RUNPATH so a relocation problem is visible immediately.
if command -v readelf >/dev/null 2>&1; then
  log ""
  log "=== recorded RUNPATH of iree-run-module ==="
  readelf -d "$IREE_RUN_MODULE" 2>/dev/null | grep -E "RPATH|RUNPATH|NEEDED.*(coral|edgetpu)" || \
    log "  (no RPATH/RUNPATH entry)"
fi
