#!/usr/bin/env bash
# Create the Python virtual environment with the pinned packages.
#
#   scripts/setup_venv.sh
#
# The toolchain's Python tools (importer, generator, comparison and hardware
# tests) run against TensorFlow/LiteRT, numpy and flatbuffers.
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

require_cmd python3

if [[ ! -d "$VENV" ]]; then
  log "=== create virtual environment ($VENV) ==="
  python3 -m venv "$VENV"
else
  log "=== virtual environment already exists ($VENV) ==="
fi

log "=== install pinned packages ==="
"$VENV/bin/pip" install --upgrade pip >/dev/null
"$VENV/bin/pip" install \
  "numpy==$PY_NUMPY" \
  "tensorflow==$PY_TENSORFLOW" \
  "flatbuffers==$PY_FLATBUFFERS" \
  "protobuf==$PY_PROTOBUF"

log ""
"$VENV/bin/python" - <<'EOF'
import flatbuffers, numpy, tensorflow
print("numpy      ", numpy.__version__)
print("tensorflow ", tensorflow.__version__)
print("flatbuffers", flatbuffers.__version__)
EOF
log ""
log "virtual environment ready at $VENV"
