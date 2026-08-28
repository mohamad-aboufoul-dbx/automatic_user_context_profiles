"""
Pure aggregation logic for the AUCP offline eval report.

This module is intentionally side-effect-free: no network calls, no SQL,
no filesystem I/O.  All inputs are ``list[dict]`` of eval_runs rows;
all outputs are plain dataclass instances.

Usage (notebook)::

    from report.aggregate import compute_report, render_report

    rows = read_eval_runs(...)          # notebook does the SQL read
    report = compute_report(rows)
    print(render_report(report))

Usage (tests)::

    from report.aggregate import compute_report, render_report

    rows = [{"task_id": "T1", "arm": "retrieved", "success": True, ...}]
    report = compute_report(rows)
    assert report.falsifiable_checks[0].verdict == "PASS"

Tie-handling (documented and enforced here):
    Primary outcome: success_rate (retrieved - placebo).
    If retrieved_rate > placebo_rate   → PASS,  basis="success-rate"
    If retrieved_rate < placebo_rate   → FLAG,  basis="success-rate"
    If retrieved_rate == placebo_rate  → efficiency tiebreak:
        Compare in priority order: exploratory_reads_before_edit,
        total_tool_calls, failed_test_cycles (lower is better).
        First metric where retrieved < placebo (strictly)  → PASS
        First metric where retrieved > placebo (strictly)  → FLAG
        All metrics equal or unavailable                   → FLAG
        Basis string records which metric (or "all-tied") decided.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All arms that appear in eval_runs
ALL_ARMS = ("empty", "static_generic", "retrieved", "placebo")

# Arms that "retrieved" is compared against
COMPARATOR_ARMS = ("empty", "static_generic", "placebo")

# Efficiency metrics (lower is better) used for tiebreak, in priority order
EFFICIENCY_TIEBREAK_METRICS = (
    "exploratory_reads_before_edit",
    "total_tool_calls",
    "failed_test_cycles",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ArmSummary:
    """Per-(task_id, arm) summary statistics.

    Medians are computed only over *successful* runs (``success=True``).
    If there are no successful runs, all median fields are ``None``.
    """
    task_id: str
    arm: str
    n_runs: int                            # total repeats observed
    n_success: int                         # how many had success=True
    success_rate: float                    # n_success / n_runs; 0.0 if n_runs==0

    # Medians among SUCCESSFUL runs only (None if no successful runs)
    med_exploratory_reads: Optional[float]
    med_total_tool_calls: Optional[float]
    med_failed_test_cycles: Optional[float]
    med_elapsed_seconds: Optional[float]
    med_tokens: Optional[float]            # input_tokens + output_tokens per run, then median


@dataclass
class PairedDelta:
    """Per-task delta: retrieved minus one comparator arm.

    Positive success_rate_delta means retrieved beat the comparator.
    Negative efficiency deltas mean retrieved used fewer resources (better).
    ``None`` values mean the metric was unavailable for one or both arms.
    """
    task_id: str
    comparator_arm: str                    # "empty" | "static_generic" | "placebo"
    success_rate_delta: Optional[float]    # retrieved_rate - comparator_rate
    med_delta_exploratory_reads: Optional[float]  # retrieved_med - comparator_med
    med_delta_tool_calls: Optional[float]
    med_delta_failed_cycles: Optional[float]
    med_delta_elapsed_seconds: Optional[float]
    med_delta_tokens: Optional[float]


@dataclass
class FalsifiableCheckResult:
    """Per-task verdict on the primary falsifiable claim.

    The claim: retrieved > placebo on success rate (with efficiency tiebreak).
    See module docstring for the full tie-handling rule.
    """
    task_id: str
    verdict: str          # "PASS" | "FLAG"
    basis: str            # how the verdict was decided (see module docstring)
    retrieved_rate: Optional[float]
    placebo_rate: Optional[float]
    detail: str           # human-readable explanation of the decision


@dataclass
class ReportData:
    """All aggregated data required to render the eval report."""
    task_ids: list[str] = field(default_factory=list)
    total_rows: int = 0
    arm_summaries: list[ArmSummary] = field(default_factory=list)
    paired_deltas: list[PairedDelta] = field(default_factory=list)
    falsifiable_checks: list[FalsifiableCheckResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_median(values: list) -> Optional[float]:
    """Return the median of non-None numeric values; None if no values."""
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return statistics.median(nums)


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Return a - b; None if either operand is None."""
    if a is None or b is None:
        return None
    return a - b


def _to_bool(v) -> bool:
    """Coerce SQL-returned booleans (may be Python bool, int 0/1, or str)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


# ---------------------------------------------------------------------------
# Step 1 — per-(task, arm) summaries
# ---------------------------------------------------------------------------

def compute_arm_summaries(
    rows: list[dict],
) -> dict[tuple[str, str], ArmSummary]:
    """Group rows by (task_id, arm) and compute summary statistics.

    Parameters
    ----------
    rows:
        eval_runs rows.  Each must contain ``task_id``, ``arm``, and
        ``success`` at minimum.  Efficiency columns may be ``None``.

    Returns
    -------
    dict mapping (task_id, arm) to ArmSummary.
    """
    # Group
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row["task_id"]), str(row["arm"]))
        grouped.setdefault(key, []).append(row)

    summaries: dict[tuple[str, str], ArmSummary] = {}
    for (task_id, arm), arm_rows in grouped.items():
        n_runs = len(arm_rows)
        successful = [r for r in arm_rows if _to_bool(r.get("success", False))]
        n_success = len(successful)
        success_rate = n_success / n_runs if n_runs > 0 else 0.0

        med_expl = _safe_median([r.get("exploratory_reads_before_edit") for r in successful])
        med_tc = _safe_median([r.get("total_tool_calls") for r in successful])
        med_fc = _safe_median([r.get("failed_test_cycles") for r in successful])
        med_sec = _safe_median([r.get("elapsed_seconds") for r in successful])

        # Combined tokens: if at least one token field is present, sum them
        token_totals: list[float] = []
        for r in successful:
            tin = r.get("input_tokens")
            tout = r.get("output_tokens")
            if tin is not None or tout is not None:
                token_totals.append(float(tin or 0) + float(tout or 0))
        med_tok = _safe_median(token_totals) if token_totals else None

        summaries[(task_id, arm)] = ArmSummary(
            task_id=task_id,
            arm=arm,
            n_runs=n_runs,
            n_success=n_success,
            success_rate=success_rate,
            med_exploratory_reads=med_expl,
            med_total_tool_calls=med_tc,
            med_failed_test_cycles=med_fc,
            med_elapsed_seconds=med_sec,
            med_tokens=med_tok,
        )

    return summaries


# ---------------------------------------------------------------------------
# Step 2 — per-task paired deltas
# ---------------------------------------------------------------------------

def compute_paired_deltas(
    summaries: dict[tuple[str, str], ArmSummary],
    task_ids: list[str],
) -> list[PairedDelta]:
    """Compute retrieved-minus-comparator deltas for every (task, comparator).

    Parameters
    ----------
    summaries:
        Output of :func:`compute_arm_summaries`.
    task_ids:
        Ordered list of task IDs to include.

    Returns
    -------
    list of PairedDelta, one per (task_id, comparator_arm).
    """
    deltas: list[PairedDelta] = []

    for task_id in task_ids:
        ret = summaries.get((task_id, "retrieved"))

        for cmp_arm in COMPARATOR_ARMS:
            cmp = summaries.get((task_id, cmp_arm))

            r_rate = ret.success_rate if ret else None
            c_rate = cmp.success_rate if cmp else None
            rate_delta = _delta(r_rate, c_rate)

            med_expl_delta = _delta(
                ret.med_exploratory_reads if ret else None,
                cmp.med_exploratory_reads if cmp else None,
            )
            med_tc_delta = _delta(
                ret.med_total_tool_calls if ret else None,
                cmp.med_total_tool_calls if cmp else None,
            )
            med_fc_delta = _delta(
                ret.med_failed_test_cycles if ret else None,
                cmp.med_failed_test_cycles if cmp else None,
            )
            med_sec_delta = _delta(
                ret.med_elapsed_seconds if ret else None,
                cmp.med_elapsed_seconds if cmp else None,
            )
            med_tok_delta = _delta(
                ret.med_tokens if ret else None,
                cmp.med_tokens if cmp else None,
            )

            deltas.append(PairedDelta(
                task_id=task_id,
                comparator_arm=cmp_arm,
                success_rate_delta=rate_delta,
                med_delta_exploratory_reads=med_expl_delta,
                med_delta_tool_calls=med_tc_delta,
                med_delta_failed_cycles=med_fc_delta,
                med_delta_elapsed_seconds=med_sec_delta,
                med_delta_tokens=med_tok_delta,
            ))

    return deltas


# ---------------------------------------------------------------------------
# Step 3 — falsifiable check (headline)
# ---------------------------------------------------------------------------

def compute_falsifiable_checks(
    summaries: dict[tuple[str, str], ArmSummary],
    task_ids: list[str],
) -> list[FalsifiableCheckResult]:
    """Compute per-task PASS/FLAG verdicts (retrieved vs placebo).

    Tie-handling rule (explicit, documented in module docstring):
    1. retrieved_rate > placebo_rate  → PASS,  basis="success-rate"
    2. retrieved_rate < placebo_rate  → FLAG,  basis="success-rate"
    3. Equal rates (tie):
       a. Iterate efficiency metrics in priority order.
       b. First strict difference: retrieved < placebo → PASS (basis=metric name);
          retrieved > placebo → FLAG (basis=metric name).
       c. All equal or None → FLAG, basis="efficiency-tiebreak (all-tied)".
    4. Missing arm data → FLAG, basis="missing-data".
    """
    checks: list[FalsifiableCheckResult] = []

    for task_id in task_ids:
        ret = summaries.get((task_id, "retrieved"))
        pla = summaries.get((task_id, "placebo"))

        r_rate: Optional[float] = ret.success_rate if ret is not None else None
        p_rate: Optional[float] = pla.success_rate if pla is not None else None

        # Missing arm data
        if ret is None or pla is None:
            missing = "retrieved" if ret is None else "placebo"
            checks.append(FalsifiableCheckResult(
                task_id=task_id,
                verdict="FLAG",
                basis="missing-data",
                retrieved_rate=r_rate,
                placebo_rate=p_rate,
                detail=(
                    f"task={task_id}: no data for {missing} arm — "
                    "cannot evaluate claim"
                ),
            ))
            continue

        # Primary: success rate
        if r_rate > p_rate:
            checks.append(FalsifiableCheckResult(
                task_id=task_id,
                verdict="PASS",
                basis="success-rate",
                retrieved_rate=r_rate,
                placebo_rate=p_rate,
                detail=(
                    f"task={task_id}: retrieved ({r_rate:.3f}) > "
                    f"placebo ({p_rate:.3f}) on success rate"
                ),
            ))
            continue

        if r_rate < p_rate:
            checks.append(FalsifiableCheckResult(
                task_id=task_id,
                verdict="FLAG",
                basis="success-rate",
                retrieved_rate=r_rate,
                placebo_rate=p_rate,
                detail=(
                    f"task={task_id}: retrieved ({r_rate:.3f}) < "
                    f"placebo ({p_rate:.3f}) on success rate — claim falsified"
                ),
            ))
            continue

        # Tie on success rate: efficiency tiebreak
        verdict: Optional[str] = None
        basis: Optional[str] = None
        detail: Optional[str] = None

        eff_vals = {
            "exploratory_reads_before_edit": (
                ret.med_exploratory_reads,
                pla.med_exploratory_reads,
            ),
            "total_tool_calls": (
                ret.med_total_tool_calls,
                pla.med_total_tool_calls,
            ),
            "failed_test_cycles": (
                ret.med_failed_test_cycles,
                pla.med_failed_test_cycles,
            ),
        }

        for metric in EFFICIENCY_TIEBREAK_METRICS:
            r_med, p_med = eff_vals[metric]
            if r_med is None or p_med is None:
                continue  # skip unavailable metric; try next
            if r_med < p_med:
                verdict = "PASS"
                basis = f"efficiency-tiebreak ({metric})"
                detail = (
                    f"task={task_id}: success-rate tie (both {r_rate:.3f}); "
                    f"retrieved {metric}={r_med} < placebo {metric}={p_med} — "
                    "retrieved leaner on efficiency"
                )
                break
            if r_med > p_med:
                verdict = "FLAG"
                basis = f"efficiency-tiebreak ({metric})"
                detail = (
                    f"task={task_id}: success-rate tie (both {r_rate:.3f}); "
                    f"retrieved {metric}={r_med} > placebo {metric}={p_med} — "
                    "retrieved worse on efficiency"
                )
                break
            # r_med == p_med: continue to next metric

        if verdict is None:
            # All metrics equal or all None
            verdict = "FLAG"
            basis = "efficiency-tiebreak (all-tied)"
            detail = (
                f"task={task_id}: success-rate tie (both {r_rate:.3f}); "
                "all efficiency tiebreak metrics are equal or unavailable — "
                "cannot distinguish, defaulting to FLAG"
            )

        checks.append(FalsifiableCheckResult(
            task_id=task_id,
            verdict=verdict,
            basis=basis,
            retrieved_rate=r_rate,
            placebo_rate=p_rate,
            detail=detail,
        ))

    return checks


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def compute_report(rows: list[dict]) -> ReportData:
    """Aggregate eval_runs rows into a full ReportData.

    Parameters
    ----------
    rows:
        ``list[dict]`` of eval_runs rows.  Each dict must contain at minimum
        ``task_id``, ``arm``, and ``success``; efficiency columns may be ``None``.
        An empty list is handled gracefully (returns an empty ReportData).

    Returns
    -------
    ReportData
        Fully populated report.  Pass to :func:`render_report` to get markdown.
    """
    if not rows:
        return ReportData(total_rows=0)

    # Preserve first-appearance order of task IDs
    seen: dict[str, None] = {}
    for row in rows:
        seen[str(row["task_id"])] = None
    task_ids = list(seen.keys())

    summaries = compute_arm_summaries(rows)
    deltas = compute_paired_deltas(summaries, task_ids)
    checks = compute_falsifiable_checks(summaries, task_ids)

    return ReportData(
        task_ids=task_ids,
        total_rows=len(rows),
        arm_summaries=list(summaries.values()),
        paired_deltas=deltas,
        falsifiable_checks=checks,
    )


# ---------------------------------------------------------------------------
# Step 4 — markdown rendering
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], decimals: int = 1) -> str:
    """Format a float for table display; '—' for None."""
    if v is None:
        return "—"
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(v)


def _fmt_delta(v: Optional[float]) -> str:
    """Format a signed delta; '—' for None."""
    if v is None:
        return "—"
    return f"{v:+.2f}"


def render_report(report: ReportData) -> str:
    """Render a ReportData to a multi-section markdown string.

    Every table and section carries the label
    "directional (N=3, one correlated feature week)".

    If the report has no data (``total_rows == 0``), a "no runs yet"
    placeholder is returned instead of crashing.
    """
    if not report.task_ids:
        return (
            "# AUCP Eval Report\n\n"
            "> **No runs yet.**  "
            "`eval_runs` is empty — run `notebooks/05_run_eval.py` first.\n"
        )

    lines: list[str] = [
        "# AUCP Eval Report",
        "",
        "> All results are **directional (N=3, one correlated feature week)**.",
        "> Not statistically significant; interpret as directional signal only.",
        "",
        f"Total rows in eval_runs: {report.total_rows}  ",
        f"Tasks: {', '.join(report.task_ids)}",
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------
    # Section 1: per-(task, arm) summary table
    # ------------------------------------------------------------------
    lines += [
        "## 1. Per-(task, arm) Summary",
        "",
        "_directional (N=3, one correlated feature week)_",
        "",
        "Medians are computed over **successful runs only** (success=True).",
        "A '—' means no successful runs in that cell.",
        "",
        "| Task | Arm | N | N_success | success_rate"
        " | med_expl_reads | med_tool_calls | med_fail_cycles"
        " | med_secs | med_tokens |",
        "|------|-----|---|-----------|-------------|----------------|----------------|-----------------|----------|------------|",
    ]

    sorted_summaries = sorted(
        report.arm_summaries,
        key=lambda s: (s.task_id, s.arm),
    )
    for s in sorted_summaries:
        lines.append(
            f"| {s.task_id} | {s.arm} | {s.n_runs} | {s.n_success}"
            f" | {s.success_rate:.2f}"
            f" | {_fmt(s.med_exploratory_reads)}"
            f" | {_fmt(s.med_total_tool_calls)}"
            f" | {_fmt(s.med_failed_test_cycles)}"
            f" | {_fmt(s.med_elapsed_seconds, 0)}"
            f" | {_fmt(s.med_tokens, 0)}"
            " |"
        )

    # ------------------------------------------------------------------
    # Section 2: paired deltas
    # ------------------------------------------------------------------
    lines += [
        "",
        "---",
        "",
        "## 2. Per-task Paired Deltas (retrieved − comparator)",
        "",
        "_directional (N=3, one correlated feature week)_",
        "",
        "Positive Δ success_rate means retrieved beat the comparator.",
        "Negative efficiency deltas mean retrieved used fewer resources (better).",
        "'—' means the metric was unavailable for one or both arms.",
        "",
        "| Task | Comparator | Δ success_rate"
        " | Δ med_expl_reads | Δ med_tool_calls | Δ med_fail_cycles"
        " | Δ med_secs | Δ med_tokens |",
        "|------|------------|----------------|------------------|------------------|------------------|------------|--------------|",
    ]

    sorted_deltas = sorted(
        report.paired_deltas,
        key=lambda d: (d.task_id, d.comparator_arm),
    )
    for d in sorted_deltas:
        lines.append(
            f"| {d.task_id} | {d.comparator_arm}"
            f" | {_fmt_delta(d.success_rate_delta)}"
            f" | {_fmt_delta(d.med_delta_exploratory_reads)}"
            f" | {_fmt_delta(d.med_delta_tool_calls)}"
            f" | {_fmt_delta(d.med_delta_failed_cycles)}"
            f" | {_fmt_delta(d.med_delta_elapsed_seconds)}"
            f" | {_fmt_delta(d.med_delta_tokens)}"
            " |"
        )

    # ------------------------------------------------------------------
    # Section 3: falsifiable check
    # ------------------------------------------------------------------
    lines += [
        "",
        "---",
        "",
        "## 3. Falsifiable Check: retrieved > placebo?",
        "",
        "_directional (N=3, one correlated feature week)_",
        "",
        "**Tie-breaking rule (explicit):**",
        "On a success-rate tie, compare efficiency metrics in priority order:",
        "`exploratory_reads_before_edit`, then `total_tool_calls`, then `failed_test_cycles`",
        "(lower is better for the agent).",
        "First metric with a strict difference decides the verdict.",
        "If all metrics are equal or unavailable → FLAG.",
        "The `basis` column records which signal decided each task.",
        "",
        "| Task | Verdict | Basis | retrieved_rate | placebo_rate | Detail |",
        "|------|---------|-------|----------------|--------------|--------|",
    ]

    for c in report.falsifiable_checks:
        r_str = f"{c.retrieved_rate:.3f}" if c.retrieved_rate is not None else "—"
        p_str = f"{c.placebo_rate:.3f}" if c.placebo_rate is not None else "—"
        lines.append(
            f"| {c.task_id} | {c.verdict} | {c.basis}"
            f" | {r_str} | {p_str} | {c.detail} |"
        )

    lines += ["", "---", ""]
    return "\n".join(lines)
