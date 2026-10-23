#!/usr/bin/env bash
# Fetch the third-party inputs the toolchain builds against.
#
#   scripts/fetch_third_party.sh [--force]
#
# Two things are needed:
#   1. the TensorFlow source tree, used only to build libtensorflow-lite.a;
#   2. libedgetpu (the header + the proprietary libedgetpu.so.1 binary) and the
#      proprietary edgetpu_compiler.
#
# The header and the compiler come from the Google Coral `edgetpu` repository;
# the runtime library comes from the `pycoral` repository.  The two are not
# interchangeable: edgetpu's own libedgetpu build targets a newer TFLite ABI
# than the 2.7 delegate in this toolchain accepts, whereas pycoral's
# libedgetpu_bin/ ships the matching binary.
#
# Nothing is vendored into this repository (all of it is large and some is
# proprietary); this script fetches it locally, exactly like IREE itself
# fetches its dependencies.
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1
require_cmd git

# Move a checkout to an exact commit, fetching it when the clone does not
# contain it.  Every external input is pinned so a bootstrap builds the verified
# revision rather than whatever the remote's default branch holds today.
checkout_pinned() { # dir, commit, url
  local dir="$1" commit="$2" url="$3"
  if ! git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
    log "  fetching pinned commit $commit"
    if ! git -C "$dir" fetch --depth 1 "$url" "$commit" 2>/dev/null; then
      git -C "$dir" fetch "$url" "$commit" || \
        die "cannot fetch $commit from $url"
    fi
  fi
  git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null || \
    die "commit $commit is not available from $url"
  if [[ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]]; then
    log "  checking out $commit"
    git -C "$dir" checkout --detach "$commit"
  fi
}

log "=== TensorFlow source ($TENSORFLOW_VERSION) ==="
if [[ -d "$TFLITE_SRC/.git" && "$FORCE" -eq 0 ]]; then
  log "  present at $TFLITE_SRC"
else
  rm -rf "$TFLITE_SRC"
  log "  cloning $TENSORFLOW_REPO (this is large)"
  # A shallow clone at the tag is enough to build the TFLite C API; the tag
  # resolves to the pinned commit, whose hash checkout_pinned verifies.
  if ! git clone --depth 1 --branch "v$TENSORFLOW_VERSION" \
        "$TENSORFLOW_REPO" "$TFLITE_SRC" 2>/dev/null; then
    log "  tag v$TENSORFLOW_VERSION not available; fetching the pinned commit"
    mkdir -p "$TFLITE_SRC"
    git -C "$TFLITE_SRC" init -q
    git -C "$TFLITE_SRC" remote add origin "$TENSORFLOW_REPO"
  fi
fi
checkout_pinned "$TFLITE_SRC" "$TENSORFLOW_COMMIT" "$TENSORFLOW_REPO"

log ""
log "=== Google Coral edgetpu (headers + edgetpu_compiler) ==="
if [[ -d "$EDGETPU_SRC/.git" && "$FORCE" -eq 0 ]]; then
  log "  present at $EDGETPU_SRC"
else
  rm -rf "$EDGETPU_SRC"
  log "  cloning $EDGETPU_REPO"
  git clone --depth 1 "$EDGETPU_REPO" "$EDGETPU_SRC"
fi
checkout_pinned "$EDGETPU_SRC" "$EDGETPU_COMMIT" "$EDGETPU_REPO"

log ""
log "=== Google Coral pycoral (the TFLite-2.7-compatible libedgetpu.so.1.0) ==="
if [[ -d "$PYCORAL_SRC/.git" && "$FORCE" -eq 0 ]]; then
  log "  present at $PYCORAL_SRC"
else
  rm -rf "$PYCORAL_SRC"
  log "  cloning $PYCORAL_REPO"
  git clone --depth 1 "$PYCORAL_REPO" "$PYCORAL_SRC"
fi
checkout_pinned "$PYCORAL_SRC" "$PYCORAL_COMMIT" "$PYCORAL_REPO"

# libedgetpu.so.1.0 lives in a platform-specific directory inside pycoral
# (libedgetpu_bin/{direct,throttled}/<platform>/); copy the build that matches
# the host architecture into the toolchain's third-party tree so build and
# runtime paths are stable.  Picking the wrong platform silently produces a
# library the linker rejects ("file in wrong format"), so the selection is
# explicit rather than left to find(1) ordering.
log ""
log "=== stage libedgetpu ==="
mkdir -p "$LIBEDGETPU_DIR"

host_platform() {
  case "$(uname -m)" in
    x86_64|amd64)  echo k8 ;;
    aarch64|arm64) echo aarch64 ;;
    armv7l|armv7)  echo armv7a ;;
    armv6l|armv6)  echo armv6 ;;
    *)             echo "" ;;
  esac
}

staged_matches_host() {
  local lib="$1"
  [[ -f "$lib" ]] || return 1
  command -v file >/dev/null 2>&1 || return 0
  local want
  case "$(uname -m)" in
    x86_64|amd64)  want="x86-64" ;;
    aarch64|arm64) want="aarch64" ;;
    armv7l|armv7)  want="ARM" ;;
    *)             return 0 ;;
  esac
  file -b "$lib" | grep -c -- "$want" >/dev/null
}

if [[ -f "$LIBEDGETPU_DIR/libedgetpu.so.1" && "$FORCE" -eq 0 ]] && staged_matches_host "$LIBEDGETPU_DIR/libedgetpu.so.1.0"; then
  log "  already staged at $LIBEDGETPU_DIR"
else
  platform="$(host_platform)"
  found=""
  if [[ -n "$platform" ]]; then
    for variant in direct throttled; do
      candidate="$PYCORAL_SRC/libedgetpu_bin/$variant/$platform/libedgetpu.so.1.0"
      if [[ -f "$candidate" ]]; then
        found="$candidate"
        break
      fi
    done
  fi
  [[ -n "$found" ]] || die "no libedgetpu.so.1.0 for host architecture '$platform' under $PYCORAL_SRC/libedgetpu_bin"
  cp "$found" "$LIBEDGETPU_DIR/libedgetpu.so.1.0"
  ln -sf libedgetpu.so.1.0 "$LIBEDGETPU_DIR/libedgetpu.so.1"
  ln -sf libedgetpu.so.1.0 "$LIBEDGETPU_DIR/libedgetpu.so"
  log "  staged $platform -> $LIBEDGETPU_DIR"
fi

log ""
log "=== stage libedgetpu headers ==="
# The headers must actually be present, not merely the directory: an earlier
# interrupted run can leave an empty directory behind.
if [[ -f "$LIBEDGETPU_HEADERS/edgetpu_c.h" && "$FORCE" -eq 0 ]]; then
  log "  already present at $LIBEDGETPU_HEADERS"
else
  mkdir -p "$LIBEDGETPU_HEADERS"
  hdr="$(find "$EDGETPU_SRC" -name 'edgetpu.h' -print -quit 2>/dev/null)"
  [[ -n "$hdr" ]] || die "edgetpu.h not found in $EDGETPU_SRC"
  src_hdr="$(dirname "$hdr")"
  cp "$src_hdr"/*.h "$LIBEDGETPU_HEADERS/"
  log "  copied headers from $src_hdr"
fi

log ""
log "=== verify edgetpu_compiler ==="
if [[ -x "$EDGETPU_SRC/compiler/x86_64/edgetpu_compiler" ]]; then
  V="$("$EDGETPU_SRC/compiler/x86_64/edgetpu_compiler" --version 2>&1 | head -1)"
  log "  $V"
else
  warn "edgetpu_compiler not found at $EDGETPU_SRC/compiler/x86_64/edgetpu_compiler"
  warn "hardware tests need it; it is redistribution-restricted and not vendored here"
fi

log ""
log "=== Eigen (for reproducible TFLite 2.7 builds) ==="
# TensorFlow 2.7 clones Eigen from gitlab.com, which intermittently rejects
# requests under load.  Fetch the same commit from the TensorFlow GCS mirror
# instead, verify the sha256 from TensorFlow's own workspace.bzl, and cache the
# archive so every TFLite configure reuses it (file://) without network access.
[[ -n "${EIGEN_URL:-}" && -n "${EIGEN_SHA256:-}" && -n "${EIGEN_COMMIT:-}" ]] || \
  die "EIGEN_URL/EIGEN_SHA256/EIGEN_COMMIT missing from versions.lock"
require_cmd curl sha256sum
archive_is_valid() {
  [[ -f "$1" ]] || return 1
  echo "${EIGEN_SHA256}  $1" | sha256sum -c - >/dev/null 2>&1
}
if archive_is_valid "$EIGEN_ARCHIVE" && [[ "$FORCE" -eq 0 ]]; then
  log "  present at $EIGEN_ARCHIVE"
else
  rm -f "$EIGEN_ARCHIVE"
  mkdir -p "$(dirname "$EIGEN_ARCHIVE")"
  tmp="$(mktemp "${EIGEN_ARCHIVE}.XXXXXX")"
  log "  downloading $EIGEN_URL"
  if ! curl -fsSL --retry 5 --retry-delay 2 -o "$tmp" "$EIGEN_URL"; then
    rm -f "$tmp"
    die "cannot download Eigen archive from $EIGEN_URL"
  fi
  log "  verifying sha256"
  if ! echo "${EIGEN_SHA256}  $tmp" | sha256sum -c - >/dev/null 2>&1; then
    rm -f "$tmp"
    die "Eigen archive sha256 mismatch (expected $EIGEN_SHA256)"
  fi
  mv "$tmp" "$EIGEN_ARCHIVE"
  log "  cached at $EIGEN_ARCHIVE"
fi

log ""
log "third-party inputs ready"
