#!/usr/bin/env bash
# Build libtensorflow-lite.a and the native Coral shim libcoral_native.so.
#
#   scripts/build_native_runtime.sh [--fresh]
#
# libcoral_native.so is the only bridge between the IREE Coral HAL loader and
# the Edge TPU: it wraps the TFLite 2.7 C API plus libedgetpu behind the small C
# interface in runtime/coral_native.h.  IREE itself never links TensorFlow.
#
# Requirements:
#   * a TensorFlow source tree at $TFLITE_SRC (see scripts/fetch_third_party.sh)
#   * libedgetpu headers + library (see scripts/fetch_third_party.sh)
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

FRESH=0
[[ "${1:-}" == "--fresh" ]] && FRESH=1

require_cmd cmake ninja clang clang++

[[ -d "$TFLITE_SRC/tensorflow/lite" ]] || \
  die "TensorFlow source not found at $TFLITE_SRC (run scripts/fetch_third_party.sh)"
[[ -f "$LIBEDGETPU_DIR/libedgetpu.so.1" ]] || \
  die "libedgetpu.so.1 not found at $LIBEDGETPU_DIR (run scripts/fetch_third_party.sh)"
[[ -f "$LIBEDGETPU_HEADERS/edgetpu_c.h" ]] || \
  die "libedgetpu headers not found at $LIBEDGETPU_HEADERS"

TFLITE_LIB="$TFLITE_BUILD/libtensorflow-lite.a"
TFLITE_DEPS=(
  "$TFLITE_BUILD/_deps/ruy-build/libruy.a"
  "$TFLITE_BUILD/_deps/flatbuffers-build/libflatbuffers.a"
  "$TFLITE_BUILD/_deps/farmhash-build/libfarmhash.a"
  "$TFLITE_BUILD/_deps/fft2d-build/libfft2d_fftsg2d.a"
  "$TFLITE_BUILD/_deps/fft2d-build/libfft2d_fftsg.a"
)

if [[ "$FRESH" -eq 1 ]]; then
  log "=== --fresh: removing $TFLITE_BUILD ==="
  rm -rf "$TFLITE_BUILD"
fi

if [[ ! -f "$TFLITE_LIB" ]]; then
  log "=== configure TFLite 2.7 ($TFLITE_BUILD) ==="
  export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" NINJAFLAGS="-j$JOBS" MAX_JOBS="$JOBS"
  # Eigen is fetched as a pinned archive (never `git clone` from GitLab).  The
  # archive is cached by scripts/fetch_third_party.sh; fall back to its URL if
  # this script is run on its own.
  [[ -n "${EIGEN_URL:-}" && -n "${EIGEN_SHA256:-}" ]] || \
    die "EIGEN_URL/EIGEN_SHA256 missing from versions.lock"
  EIGEN_MODULE="$TFLITE_SRC/tensorflow/lite/tools/cmake/modules/eigen.cmake"
  if [[ -f "$EIGEN_MODULE" ]] && \
     ! grep -q "GIT_TAG $EIGEN_COMMIT" "$EIGEN_MODULE"; then
    die "EIGEN_COMMIT=$EIGEN_COMMIT does not match $EIGEN_MODULE"
  fi
  eigen_url="$EIGEN_URL"
  if [[ -f "$EIGEN_ARCHIVE" ]] && \
     echo "${EIGEN_SHA256}  $EIGEN_ARCHIVE" | sha256sum -c - >/dev/null 2>&1; then
    eigen_url="file://$EIGEN_ARCHIVE"
  fi
  log "  Eigen: $eigen_url"
  eigen_args=(
    "-DOVERRIDABLE_FETCH_CONTENT_eigen_URL=$eigen_url"
    "-DOVERRIDABLE_FETCH_CONTENT_eigen_URL_HASH=SHA256=$EIGEN_SHA256"
  )
  # TFLite 2.7 relies on transitive <cstdint>/<limits> includes that newer
  # clang no longer provides; force-include them. Harmless elsewhere.
  cmake -S "$TFLITE_SRC/tensorflow/lite" -B "$TFLITE_BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_FLAGS="-include cstdint -include limits -Wno-macro-redefined -Wno-error=macro-redefined" \
    -DTFLITE_ENABLE_XNNPACK=OFF \
    -DTFLITE_ENABLE_INSTALL=OFF \
    "${eigen_args[@]}"

  log ""
  log "=== build tensorflow-lite (-j$JOBS) ==="
  cmake --build "$TFLITE_BUILD" --target tensorflow-lite --parallel "$JOBS"
fi
# The _deps archives are produced by the same build.
for f in "${TFLITE_DEPS[@]}"; do
  [[ -f "$f" ]] || die "expected TFLite dependency archive missing: $f"
done
require_file "$TFLITE_LIB"
log "libtensorflow-lite.a ready"

log ""
log "=== build libcoral_native.so ==="
mkdir -p "$(dirname "$CORAL_NATIVE_SO")"
# The TFLite C API objects reference the _deps archives only transitively, so
# group the whole set to keep the linker from dropping symbols. The RUNPATH is
# $ORIGIN so the shim finds libedgetpu.so.1 wherever the two are installed
# together (the runtime build copies them beside iree-run-module).
clang -shared -fPIC -O2 -std=gnu11 \
  -I"$TFLITE_SRC" -I"$LIBEDGETPU_HEADERS" \
  "$TOOLCHAIN_ROOT/runtime/libcoral_native.c" \
  -Wl,--start-group "$TFLITE_LIB" "${TFLITE_DEPS[@]}" -Wl,--end-group \
  "$LIBEDGETPU_DIR/libedgetpu.so.1" \
  -lstdc++ -lm -ldl -lpthread -lrt \
  -Wl,-soname,libcoral_native.so \
  -Wl,-rpath,'$ORIGIN' \
  -o "$CORAL_NATIVE_SO"

require_file "$CORAL_NATIVE_SO"
log "built $CORAL_NATIVE_SO"
# Stage libedgetpu beside the shim so the two can be installed as a pair.
cp -f "$LIBEDGETPU_DIR/libedgetpu.so.1" "$(dirname "$CORAL_NATIVE_SO")/libedgetpu.so.1"
ln -sf libedgetpu.so.1 "$(dirname "$CORAL_NATIVE_SO")/libedgetpu.so"
log "staged libedgetpu.so.1 beside the shim"
