#!/usr/bin/env python3
"""Validate a Databricks `jobs get-run-output` JSON carries the expected sentinel.

Usage:
    databricks jobs get-run-output <task_run_id> -o json \\
      | python3 _check_sentinel.py <expected_sentinel>

Reads the full get-run-output JSON from STDIN (a plain pipe — no heredoc, so
stdin is never shadowed) and the expected sentinel string from argv[1].

Checks that notebook_output.result exists, parses as JSON, and that its
"sentinel" field equals <expected_sentinel>. A notebook that stopped early
(e.g. only its import cell ran) never calls dbutils.notebook.exit, so
notebook_output.result is absent and validation fails.

Exit 0 on match; exit 1 on any mismatch/missing/parse failure (diagnostics to
stderr); exit 2 on usage error.
"""
import json
import sys


def check(raw_output: str, expected: str) -> tuple[bool, str]:
    """Return (ok, message). Pure function so it is unit-testable."""
    if not raw_output.strip():
        return False, "get-run-output produced empty stdout (no JSON to parse)"
    try:
        d = json.loads(raw_output)
    except Exception as exc:  # noqa: BLE001 - report any parse failure
        return False, f"get-run-output is not valid JSON: {exc}"

    result = (d.get("notebook_output") or {}).get("result")
    if not result:
        return False, (
            "notebook_output.result missing/empty — the notebook did not reach "
            "dbutils.notebook.exit (incomplete execution)"
        )
    try:
        sentinel = json.loads(result).get("sentinel", "")
    except Exception as exc:  # noqa: BLE001
        return False, f"notebook_output.result is not valid JSON: {exc} (raw={result[:200]!r})"

    if sentinel == expected:
        return True, sentinel
    return False, f"expected sentinel {expected!r}, got {sentinel!r} (raw={result[:200]!r})"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: _check_sentinel.py <expected_sentinel>", file=sys.stderr)
        return 2
    expected = argv[1]
    raw = sys.stdin.read()
    ok, msg = check(raw, expected)
    if ok:
        print(f"    sentinel OK: {msg}")
        return 0
    print(f"    SENTINEL FAIL: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
