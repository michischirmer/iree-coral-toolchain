#!/usr/bin/env bash
# Run every toolchain test suite sequentially and report a summary.
#
#   scripts/run_all_tests.sh                     # every suite
#   scripts/run_all_tests.sh test_mobilenet_v2_quant.py
#
# Suites share the generated/ artifact directory, so they run one at a time.
# Hardware suites are skipped with a clear note when /dev/apex_0 or the
# proprietary edgetpu_compiler is absent, so the same driver works on a
# build-only machine.
set -uo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

LOGDIR="$WORKSPACE/test-logs"
mkdir -p "$LOGDIR"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

DEFAULT_SUITES=(
  test_mobilenet_v2_quant.py
  test_mobilenet_v2_hardware.py
)

SUITES=("$@")
if [ ${#SUITES[@]} -eq 0 ]; then
  SUITES=("${DEFAULT_SUITES[@]}")
fi

[[ -x "$PY" ]] || die "python environment missing (run scripts/setup_venv.sh)"

RC=0
printf '\n%-40s %-8s %-7s %s\n' "SUITE" "RESULT" "TIME" "LOG"
printf '%s\n' "--------------------------------------------------------------------------------"
for suite in "${SUITES[@]}"; do
  name="$(basename "$suite")"
  log_file="$LOGDIR/${name%.py}.log"

  # Skip hardware-only suites when the prerequisites are missing.
  if [[ "$name" == "test_mobilenet_v2_hardware.py" ]]; then
    if [[ ! -e /dev/apex_0 ]]; then
      printf '%-40s %-8s %-7s %s\n' "$name" "SKIP" "-" "no /dev/apex_0"
      continue
    fi
    if [[ ! -x "$EDGETPU_COMPILER" ]]; then
      printf '%-40s %-8s %-7s %s\n' "$name" "SKIP" "-" "no edgetpu_compiler"
      continue
    fi
  fi

  start=$(date +%s)
  if "$PY" "$TOOLCHAIN_ROOT/tests/$name" >"$log_file" 2>&1; then
    result="PASS"
  else
    result="FAIL"
    RC=1
  fi
  elapsed=$(( $(date +%s) - start ))
  printf '%-40s %-8s %-7s %s\n' "$name" "$result" "${elapsed}s" "$log_file"
  if [[ "$result" == "FAIL" ]]; then
    sed -n '1,40p' "$log_file" | sed 's/^/    /'
  fi
done
printf '%s\n' "--------------------------------------------------------------------------------"
if [[ "$RC" -eq 0 ]]; then
  log "ALL SUITES PASSED"
else
  log "SOME SUITES FAILED (see $LOGDIR)"
fi
exit "$RC"
