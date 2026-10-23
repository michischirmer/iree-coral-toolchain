#!/usr/bin/env bash
# Fetch the test data (models and images) used by the toolchain tests.
#
#   scripts/fetch_test_data.sh [--force]
#
# The models come from the Google Coral `edgetpu` repository, which
# scripts/fetch_third_party.sh already clones.  The images ship with the same
# repository.  Test data is not vendored because the model is ~3.4 MiB and
# these files are upstream artifacts.
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

DEST="$TOOLCHAIN_ROOT/tests/test_data"
MODEL="mobilenet_v2_1.0_224_quant.tflite"
IMAGES=(cat.bmp grace_hopper.bmp bird.bmp owl.jpg parrot.jpg)

mkdir -p "$DEST"

if [[ ! -d "$EDGETPU_SRC" ]]; then
  log "=== edgetpu checkout missing; fetching it ==="
  "$TOOLCHAIN_ROOT/scripts/fetch_third_party.sh"
fi

SRC_TEST_DATA="$EDGETPU_SRC/test_data"
[[ -d "$SRC_TEST_DATA" ]] || die "test data not found at $SRC_TEST_DATA"

log "=== copy test data -> $DEST ==="
copy_one() {
  local name="$1"
  local src="$SRC_TEST_DATA/$name"
  [[ -f "$src" ]] || die "missing upstream test file: $src"
  if [[ -f "$DEST/$name" && "$FORCE" -eq 0 ]]; then
    log "  have     $name"
  else
    cp "$src" "$DEST/$name"
    log "  copied   $name"
  fi
}

copy_one "$MODEL"
for img in "${IMAGES[@]}"; do
  copy_one "$img"
done

log ""
log "=== verify sha256 (MobileNetV2) ==="
HAVE="$(sha256sum "$DEST/$MODEL" | awk '{print $1}')"
if [[ "$HAVE" == "$TEST_MODEL_SHA256" ]]; then
  log "  OK       $MODEL ($HAVE)"
else
  warn "sha256 mismatch for $MODEL"
  warn "  expected $TEST_MODEL_SHA256"
  warn "  got      $HAVE"
  die "test data does not match the pinned revision (re-run scripts/fetch_third_party.sh --force)"
fi

log ""
log "test data ready in $DEST"
