#!/usr/bin/env python3
"""
05_run_eval.py — LOCAL entrypoint for the AUCP offline eval matrix.

NOT submitted via ``databricks jobs submit``.  Run locally:

    cd /path/to/automatic_user_context_profiles
    python notebooks/05_run_eval.py [options]

The driver executes one fixed Claude-Code agent per (task × arm × repeat),
reads frozen artifacts from memory_artifacts (UC) via the SQL statement-
execution API, and writes one eval_runs row per run plus uploads transcript +
diff to the raw Volume.

Usage
-----
    python notebooks/05_run_eval.py \\
        [--tasks T1 T2 T3] \\
        [--arms empty static_generic retrieved placebo] \\
        [--repeats 2] \\
        [--model databricks-claude-sonnet-4-5] \\
        [--warehouse 41659c95dacd3bf0] \\
        [--profile hackathon] \\
        [--platform-repo ~/Projects/platform] \\
        [--dry-run]

Options
-------
--tasks         Space-separated task IDs to run (default: T1 T2 T3).
--arms          Space-separated arms (default: empty static_generic retrieved placebo).
--repeats       Repeats per (task, arm) (default: 2).
--model         Model alias passed to --model in the Claude Code CLI.
                EMPIRICAL: confirm the exact value at Task 5.3.
                Default: databricks-claude-sonnet-4-5.
--warehouse     Databricks SQL warehouse ID (default: 41659c95dacd3bf0).
--profile       ~/.databrickscfg profile (default: hackathon).
--platform-repo Absolute path to ~/Projects/platform (default: ~/Projects/platform).
--dry-run       Print what would run without launching any agents or writing to UC.

Architecture notes
------------------
- Outputs only go to UC:
    eval_runs row   → SQL statement-execution API (INSERT)
    transcript.jsonl, final.diff → Volume at raw/eval_runs/{run_id}/
- See src/harness/run.py for the seam design and per-run orchestration.
- The live matrix is NOT executed here in testing; see task-5.3 for the
  confirmed --model value and stream-json field paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure src/ is on the path when run from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from harness.run import (
    DEFAULT_MODEL,
    DEFAULT_PLATFORM_REPO,
    DEFAULT_PROFILE,
    DEFAULT_WAREHOUSE_ID,
    RUNS_VOLUME_BASE,
    run_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AUCP offline eval matrix runner (local entrypoint)."
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["T1", "T2", "T3"],
        metavar="TASK_ID",
        help="Task IDs to run.",
    )
    parser.add_argument(
        "--arms", nargs="+",
        default=["empty", "static_generic", "retrieved", "placebo"],
        metavar="ARM",
        help="Arms to run.",
    )
    parser.add_argument(
        "--repeats", type=int, default=2,
        help="Number of repeats per (task, arm).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=(
            "Model alias for --model in the Claude Code CLI. "
            "EMPIRICAL: confirm the exact value at Task 5.3. "
            f"Default: {DEFAULT_MODEL}"
        ),
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
        "--platform-repo", default=DEFAULT_PLATFORM_REPO,
        dest="platform_repo",
        help=f"Absolute path to the platform repo. Default: {DEFAULT_PLATFORM_REPO}",
    )
    parser.add_argument(
        "--eval-tasks-dir",
        default=os.path.join(os.path.dirname(_HERE), "eval", "tasks"),
        dest="eval_tasks_dir",
        help="Directory containing T*.json task definitions.",
    )
    parser.add_argument(
        "--runs-volume-base",
        default=RUNS_VOLUME_BASE,
        dest="runs_volume_base",
        help=f"Volume base path for run artifacts. Default: {RUNS_VOLUME_BASE}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the run matrix without launching agents or writing to UC.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n_runs = len(args.tasks) * len(args.arms) * args.repeats
    print(f"AUCP eval matrix: {len(args.tasks)} tasks × {len(args.arms)} arms × "
          f"{args.repeats} repeats = {n_runs} total runs")
    print(f"  tasks    : {args.tasks}")
    print(f"  arms     : {args.arms}")
    print(f"  model    : {args.model}")
    print(f"  warehouse: {args.warehouse}")
    print(f"  profile  : {args.profile}")
    print(f"  platform : {args.platform_repo}")
    print(f"  tasks dir: {args.eval_tasks_dir}")

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for tid in args.tasks:
            for arm in args.arms:
                for rep in range(1, args.repeats + 1):
                    print(f"  run_one(task={tid}, arm={arm}, repeat={rep})")
        print("\n[DRY RUN] No agents launched, no UC writes performed.")
        return

    print("\nStarting eval matrix...", flush=True)
    rows = run_matrix(
        task_ids=args.tasks,
        arms=args.arms,
        repeats=args.repeats,
        eval_tasks_dir=args.eval_tasks_dir,
        model=args.model,
        platform_repo=args.platform_repo,
        warehouse_id=args.warehouse,
        profile=args.profile,
        runs_volume_base=args.runs_volume_base,
    )

    print(f"\n{'─' * 60}")
    print(f"Matrix complete: {len(rows)} runs")
    successes = sum(1 for r in rows if r.success)
    failures = sum(1 for r in rows if not r.success)
    print(f"  success : {successes}")
    print(f"  failure : {failures}")
    print(f"{'─' * 60}")

    for row in rows:
        status = "PASS" if row.success else "FAIL"
        reason = f" [{row.failure_reason}]" if row.failure_reason else ""
        print(
            f"  {status}  {row.task_id}/{row.arm} rep={row.repeat_number}"
            f"  run_id={row.run_id}{reason}"
        )

    print(f"\nAll rows written to eval_runs in UC namespace "
          f"ai_fde_hackathon_catalog.automatic_user_context_profiles")


if __name__ == "__main__":
    main()
