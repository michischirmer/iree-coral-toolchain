#!/usr/bin/env bash
# Build the Coral loader negative-test binary.
#
#   scripts/build_negative_test.sh
#
# This binary drives the regular IREE runtime and confirms the Coral loader
# rejects malformed or foreign executables rather than crashing or silently
# falling back.  It links against the runtime build tree (which contains the
# Coral loader when configured with -DIREE_HAL_EXECUTABLE_LOADER_CORAL=ON) plus
# the native shim.
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

require_cmd clang
[[ -f "$CORAL_NATIVE_SO" ]] || \
  die "libcoral_native.so not found (run scripts/build_native_runtime.sh first)"
[[ -d "$RUNTIME_INSTALL_BUILD" ]] || \
  die "runtime build tree not found (run scripts/build_runtime.sh first)"
[[ -f "$LIBEDGETPU_DIR/libedgetpu.so.1" ]] || \
  die "libedgetpu.so.1 not found at $LIBEDGETPU_DIR"

OUT="${1:-$TOOLCHAIN_ROOT/runtime/coral_loader_negative_test}"

# The runtime build tree holds the compiled IREE runtime; link it statically.
# The archives are wrapped in --start-group/--end-group so the static link is
# independent of the order `find` happens to return them in (the Coral loader
# and the IREE base library reference each other). The RUNPATH is $ORIGIN so
# the binary finds libcoral_native.so (and, through it, libedgetpu.so.1)
# beside itself wherever the pair is installed.
clang -std=gnu11 -O1 \
  -I"$IREE_DIR" -I"$IREE_DIR/runtime/src" -I"$TOOLCHAIN_ROOT/runtime" \
  -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl \
  "$TOOLCHAIN_ROOT/runtime/coral_loader_negative_test.c" \
  -Wl,--start-group $(find "$RUNTIME_INSTALL_BUILD" -name '*.a' | tr '\n' ' ') -Wl,--end-group \
  -L"$(dirname "$CORAL_NATIVE_SO")" -lcoral_native \
  -L"$LIBEDGETPU_DIR" -Wl,-rpath-link,"$LIBEDGETPU_DIR" \
  -Wl,-rpath,'$ORIGIN' \
  -lstdc++ -lm -ldl -lpthread -lrt \
  -o "$OUT"

log "built $OUT"
