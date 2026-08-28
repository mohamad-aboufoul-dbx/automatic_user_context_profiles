#!/usr/bin/env python3
"""
06_report.py — LOCAL entrypoint for the AUCP offline eval report.

NOT submitted via ``databricks jobs submit``.  Run locally:

    cd /path/to/automatic_user_context_profiles
    python notebooks/06_report.py [options]

Reads ``eval_runs`` from UC via the SQL statement-execution API
(same local read path the controller uses in notebooks/05_run_eval.py)
and renders a markdown report with per-task success rates, medians,
and paired deltas.

Every table is labeled "directional (N=3, one correlated feature week)"
as required by SPEC §9 and task-6.1-brief.

Architecture
------------
All aggregation logic lives in the pure, testable module
``src/report/aggregate.py``.  This notebook is a thin wrapper::

    read eval_runs rows (SQL statement-execution API)
          │
          ▼
    compute_report(rows)          ← src/report/aggregate.py
          │
          ▼
    render_report(report)         ← src/report/aggregate.py
          │
          ▼
    print / write to file

The SQL read is the only I/O here; tests bypass it by calling
``compute_report()`` directly with synthetic row lists.

Usage
-----
    python notebooks/06_report.py \\
        [--warehouse 41659c95dacd3bf0] \\
        [--profile hackathon] \\
        [--output report.md]           # optional; defaults to stdout

Options
-------
--warehouse   Databricks SQL warehouse ID (default: 41659c95dacd3bf0).
--profile     ~/.databrickscfg profile (default: hackathon).
--output      Path to write the markdown report (default: stdout only).
--charts-dir  Directory to write optional matplotlib PNG charts (opt-in; omit to skip).
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure src/ is on the path when run from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from report.aggregate import compute_report, render_report

# ---------------------------------------------------------------------------
# Constants (mirrors defaults in notebooks/05_run_eval.py)
# ---------------------------------------------------------------------------

CATALOG = "ai_fde_hackathon_catalog"
SCHEMA = "automatic_user_context_profiles"
EVAL_RUNS_TABLE = f"{CATALOG}.{SCHEMA}.eval_runs"

DEFAULT_WAREHOUSE_ID = "41659c95dacd3bf0"
DEFAULT_PROFILE = "hackathon"

_EVAL_RUNS_COLUMNS = (
    "task_id",
    "arm",
    "repeat_number",
    "success",
    "acceptance_tests_passed",
    "regression_tests_passed",
    "forbidden_files_unchanged",
    "exploratory_reads_before_edit",
    "total_tool_calls",
    "failed_test_cycles",
    "elapsed_seconds",
    "input_tokens",
    "output_tokens",
)


# ---------------------------------------------------------------------------
# SQL read (the only I/O in this notebook)
# ---------------------------------------------------------------------------

def read_eval_runs(warehouse_id: str, profile: str) -> list[dict]:
    """Query all eval_runs rows via the SQL statement-execution API.

    Returns a ``list[dict]`` where each dict has the columns in
    ``_EVAL_RUNS_COLUMNS``, with Python-native types (bool, int, float, None).

    Raises ``RuntimeError`` if the query fails.
    """
    # Lazy import — unit tests never reach this function
    from databricks.sdk import WorkspaceClient  # type: ignore
    from databricks.sdk.service.sql import StatementState  # type: ignore

    w = WorkspaceClient(profile=profile)
    cols = ", ".join(_EVAL_RUNS_COLUMNS)
    sql = (
        f"SELECT {cols} FROM {EVAL_RUNS_TABLE}"
        " ORDER BY task_id, arm, repeat_number"
    )
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="60s",
    )
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(
            f"eval_runs SELECT failed: {resp.status.error}"
        )

    col_names = [c.name for c in (resp.manifest.schema.columns or [])]
    rows: list[dict] = []
    for raw_row in (resp.result.data_array or []):
        rd = dict(zip(col_names, raw_row))
        rows.append(_coerce_row(rd))
    return rows


def _coerce_row(rd: dict) -> dict:
    """Convert SQL data_array values to Python-native types.

    The Databricks SDK may return all values as strings; this function
    normalises the types that the aggregator expects.
    """
    bool_cols = {
        "success",
        "acceptance_tests_passed",
        "regression_tests_passed",
        "forbidden_files_unchanged",
    }
    int_cols = {
        "repeat_number",
        "exploratory_reads_before_edit",
        "total_tool_calls",
        "failed_test_cycles",
        "input_tokens",
        "output_tokens",
    }
    float_cols = {"elapsed_seconds"}

    out: dict = {}
    for k, v in rd.items():
        if v is None or v == "null":
            out[k] = None
        elif k in bool_cols:
            if isinstance(v, bool):
                out[k] = v
            elif isinstance(v, int):
                out[k] = bool(v)
            else:
                out[k] = str(v).strip().lower() in ("true", "1", "yes")
        elif k in int_cols:
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                out[k] = None
        elif k in float_cols:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = None
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Optional: matplotlib chart generation (lazy import — skipped in tests)
# ---------------------------------------------------------------------------

def _try_save_charts(report, output_dir: str) -> list[str]:
    """Generate and save directional bar charts as PNGs (optional).

    Returns a list of file paths that were written.
    Silently returns [] if matplotlib is not installed or chart generation fails.
    Dependency is intentionally imported lazily so this module remains importable
    without matplotlib.
    """
    try:
        import matplotlib  # type: ignore  # noqa: F401
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return []

    saved: list[str] = []
    if not report.task_ids:
        return saved

    try:
        # Chart: success rate per (task, arm)
        from collections import defaultdict
        task_arm_rate: dict[str, dict[str, float]] = defaultdict(dict)
        for s in report.arm_summaries:
            task_arm_rate[s.task_id][s.arm] = s.success_rate

        arms_order = ["empty", "static_generic", "retrieved", "placebo"]
        n_tasks = len(report.task_ids)
        n_arms = len(arms_order)
        x = range(n_tasks)

        fig, ax = plt.subplots(figsize=(max(6, n_tasks * 2), 4))
        bar_width = 0.2
        for i, arm in enumerate(arms_order):
            rates = [
                task_arm_rate.get(tid, {}).get(arm, 0.0)
                for tid in report.task_ids
            ]
            offsets = [v + (i - n_arms / 2 + 0.5) * bar_width for v in x]
            ax.bar(offsets, rates, bar_width, label=arm)

        ax.set_xticks(list(x))
        ax.set_xticklabels(report.task_ids)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Success rate")
        ax.set_title(
            "Success rate per (task, arm)\n"
            "directional (N=3, one correlated feature week)"
        )
        ax.legend()
        fig.tight_layout()

        chart_path = os.path.join(output_dir, "chart_success_rate.png")
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)
        saved.append(chart_path)

    except Exception as exc:  # noqa: BLE001
        print(f"[06_report] chart generation skipped: {exc}", flush=True)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AUCP offline eval report (local entrypoint)."
    )
    parser.add_argument(
        "--warehouse", default=DEFAULT_WAREHOUSE_ID,
        help=f"Databricks SQL warehouse ID. Default: {DEFAULT_WAREHOUSE_ID}",
    )
    parser.add_argument(
        "--profile", default=DEFAULT_PROFILE,
        help=f"~/.databrickscfg profile. Default: {DEFAULT_PROFILE}",
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Write markdown report to this file (default: stdout only).",
    )
    parser.add_argument(
        "--charts-dir", default=None, metavar="DIR",
        help=(
            "Directory to save optional matplotlib PNGs. "
            "Omit to skip chart generation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        f"[06_report] Reading eval_runs from {EVAL_RUNS_TABLE} "
        f"(warehouse={args.warehouse}, profile={args.profile})",
        flush=True,
    )

    rows = read_eval_runs(args.warehouse, args.profile)
    print(f"[06_report] {len(rows)} rows read from eval_runs.", flush=True)

    report = compute_report(rows)

    if not report.task_ids:
        print("[06_report] eval_runs is empty — no runs to report yet.", flush=True)

    md = render_report(report)

    # Always print to stdout
    print("\n" + "=" * 72)
    print(md)
    print("=" * 72 + "\n")

    # Optionally write to file
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"[06_report] Report written to {args.output}", flush=True)

    # Optional charts
    if args.charts_dir:
        os.makedirs(args.charts_dir, exist_ok=True)
        saved = _try_save_charts(report, args.charts_dir)
        for p in saved:
            print(f"[06_report] Chart saved: {p}", flush=True)
        if not saved:
            print(
                "[06_report] No charts generated "
                "(matplotlib not installed or no data).",
                flush=True,
            )


if __name__ == "__main__":
    main()
