#!/usr/bin/env bash
# Local test for _check_sentinel.py — exercises the SAME invocation pattern the
# runner uses (representative get-run-output JSON piped into the parser on
# stdin). Guards against the JSONDecodeError regression where the parser saw
# empty stdin. No Databricks connection required.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CHECK="$HERE/_check_sentinel.py"
fail=0

# run_case <name> <expected_sentinel> <get-run-output-json> <want_rc>
run_case() {
    local name="$1" expected="$2" json="$3" want_rc="$4"
    printf '%s' "$json" | python3 "$CHECK" "$expected" >/dev/null 2>&1
    local rc=$?
    if [ "$rc" -eq "$want_rc" ]; then
        echo "  PASS: $name (rc=$rc)"
    else
        echo "  FAIL: $name (rc=$rc, want=$want_rc)"
        fail=1
    fi
}

echo "=== _check_sentinel.py tests ==="

# The exact live payload shape that previously triggered JSONDecodeError:
# valid get-run-output with a valid nested sentinel result -> must PASS.
run_case "valid setup sentinel (live shape)" "setup:OK" \
  '{"notebook_output": {"result": "{\"sentinel\": \"setup:OK\"}"}, "metadata": {}}' 0

# Ingest sentinel with extra summary fields -> PASS.
run_case "valid ingest sentinel w/ counts" "ingest:OK" \
  '{"notebook_output": {"result": "{\"sentinel\": \"ingest:OK\", \"rows_total\": 247, \"rows_user\": 247}"}}' 0

# Sessionize sentinel -> PASS.
run_case "valid sessionize sentinel" "sessionize:OK" \
  '{"notebook_output": {"result": "{\"sentinel\": \"sessionize:OK\", \"sessions\": 110}"}}' 0

# Wrong sentinel value -> FAIL.
run_case "wrong sentinel" "sessionize:OK" \
  '{"notebook_output": {"result": "{\"sentinel\": \"setup:OK\"}"}}' 1

# notebook_output.result missing -> notebook stopped early -> FAIL (this is the
# incomplete-execution class the sentinel exists to catch).
run_case "missing result (early stop)" "setup:OK" \
  '{"notebook_output": {}, "metadata": {}}' 1

# Empty top-level object -> FAIL.
run_case "empty output object" "setup:OK" '{}' 1

# result present but not JSON -> FAIL.
run_case "result not json" "setup:OK" \
  '{"notebook_output": {"result": "traceback..."}}' 1

# Empty stdin -> FAIL cleanly (the precise regression: piped JSON never arrived).
run_case "empty stdin (regression)" "setup:OK" '' 1

if [ "$fail" -eq 0 ]; then
    echo "ALL SENTINEL TESTS PASSED"
else
    echo "SENTINEL TESTS FAILED"
    exit 1
fi
