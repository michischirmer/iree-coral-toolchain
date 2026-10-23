#!/usr/bin/env bash
# Build the Coral-enabled IREE compiler (`iree-compile`) from the fork.
#
#   scripts/build_compiler.sh [--fresh]
#
# Produces $COMPILER_BUILD/tools/iree-compile.  Incremental by default so
# repeated runs after a source edit are cheap; --fresh wipes the build tree.
#
# The configuration is deliberately minimal: compiler-only, only the Coral and
# VMVX targets, only the local-sync/local-task HAL drivers.
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

FRESH=0
[[ "${1:-}" == "--fresh" ]] && FRESH=1

require_cmd cmake ninja clang clang++ git
[[ -d "$IREE_DIR" ]] || die "IREE fork not found at $IREE_DIR (run scripts/bootstrap_iree.sh)"

if [[ "$FRESH" -eq 1 ]]; then
  log "=== --fresh: removing $COMPILER_BUILD ==="
  rm -rf "$COMPILER_BUILD"
fi

log "=== configure ($COMPILER_BUILD) ==="
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" NINJAFLAGS="-j$JOBS" MAX_JOBS="$JOBS"
cmake -S "$IREE_DIR" -B "$COMPILER_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_RPATH="\$ORIGIN;\$ORIGIN/../lib" \
  -DCMAKE_INSTALL_RPATH="\$ORIGIN;\$ORIGIN/../lib" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF \
  -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_BENCHMARKS=OFF \
  -DIREE_BUILD_COMPILER=ON -DIREE_BUILD_RUNTIME=OFF \
  -DIREE_INPUT_TORCH=OFF \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_CORAL=ON \
  -DIREE_TARGET_BACKEND_VMVX=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON

log ""
log "=== build iree-compile (-j$JOBS) ==="
cmake --build "$COMPILER_BUILD" --target iree-compile --parallel "$JOBS"

require_file "$IREE_COMPILE"
log ""
log "built $IREE_COMPILE"
"$IREE_COMPILE" --version 2>/dev/null || true
