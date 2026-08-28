"""
Unit tests for src/report/aggregate.py — Task 6.1.

All tests are offline: no SQL calls, no network, no filesystem I/O.
Row lists are hand-constructed with hand-computed expected values.

Coverage:
  - success rate math (0/N, K/N, N/N)
  - median-among-successes (with and without successes; mixed success/fail)
  - combined-token median
  - paired delta computation
  - falsifiable check:
      (a) retrieved clearly > placebo → PASS, basis=success-rate
      (b) retrieved < placebo → FLAG, basis=success-rate
      (c) rate tie broken by efficiency — retrieved wins → PASS
      (d) rate tie broken by efficiency — retrieved loses → FLAG
      (e) rate tie, first tiebreak metric equal, second metric decides
      (f) all tiebreak metrics equal → FLAG, basis=all-tied
      (g) missing arm data → FLAG, basis=missing-data
  - empty input → ReportData with all lists empty, total_rows=0
  - render_report produces "no runs yet" for empty input
  - render_report produces all three section headers for non-empty input
"""

from __future__ import annotations

from typing import Optional

import pytest

from report.aggregate import (
    ArmSummary,
    FalsifiableCheckResult,
    PairedDelta,
    ReportData,
    _delta,
    _safe_median,
    _to_bool,
    compute_arm_summaries,
    compute_falsifiable_checks,
    compute_paired_deltas,
    compute_report,
    render_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(
    task_id: str = "T1",
    arm: str = "retrieved",
    repeat_number: int = 1,
    success: bool = True,
    exploratory_reads_before_edit: Optional[int] = None,
    total_tool_calls: Optional[int] = None,
    failed_test_cycles: Optional[int] = None,
    elapsed_seconds: Optional[float] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    acceptance_tests_passed: bool = True,
    regression_tests_passed: Optional[bool] = None,
    forbidden_files_unchanged: bool = True,
) -> dict:
    """Construct a minimal eval_runs row dict."""
    return {
        "task_id": task_id,
        "arm": arm,
        "repeat_number": repeat_number,
        "success": success,
        "acceptance_tests_passed": acceptance_tests_passed,
        "regression_tests_passed": regression_tests_passed,
        "forbidden_files_unchanged": forbidden_files_unchanged,
        "exploratory_reads_before_edit": exploratory_reads_before_edit,
        "total_tool_calls": total_tool_calls,
        "failed_test_cycles": failed_test_cycles,
        "elapsed_seconds": elapsed_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# ---------------------------------------------------------------------------
# _to_bool
# ---------------------------------------------------------------------------

class TestToBool:
    def test_true_bool(self):
        assert _to_bool(True) is True

    def test_false_bool(self):
        assert _to_bool(False) is False

    def test_int_one(self):
        assert _to_bool(1) is True

    def test_int_zero(self):
        assert _to_bool(0) is False

    def test_string_true(self):
        assert _to_bool("true") is True
        assert _to_bool("True") is True

    def test_string_false_is_false(self):
        # "false" must NOT be truthy — critical distinction vs plain bool("false")
        assert _to_bool("false") is False
        assert _to_bool("False") is False

    def test_string_zero(self):
        assert _to_bool("0") is False


# ---------------------------------------------------------------------------
# _safe_median
# ---------------------------------------------------------------------------

class TestSafeMedian:
    def test_empty_list_returns_none(self):
        assert _safe_median([]) is None

    def test_all_none_returns_none(self):
        assert _safe_median([None, None]) is None

    def test_single_value(self):
        assert _safe_median([5]) == 5.0

    def test_odd_count(self):
        assert _safe_median([1, 3, 5]) == 3.0

    def test_even_count_averages_middle_two(self):
        assert _safe_median([1, 2, 3, 4]) == 2.5

    def test_none_values_are_filtered(self):
        # [1, None, 3] → median([1, 3]) = 2.0
        assert _safe_median([1, None, 3]) == 2.0


# ---------------------------------------------------------------------------
# _delta
# ---------------------------------------------------------------------------

class TestDelta:
    def test_both_present(self):
        assert _delta(3.0, 2.0) == pytest.approx(1.0)

    def test_negative_delta(self):
        assert _delta(1.0, 3.0) == pytest.approx(-2.0)

    def test_a_none(self):
        assert _delta(None, 2.0) is None

    def test_b_none(self):
        assert _delta(1.0, None) is None

    def test_both_none(self):
        assert _delta(None, None) is None


# ---------------------------------------------------------------------------
# compute_arm_summaries
# ---------------------------------------------------------------------------

class TestComputeArmSummaries:

    def test_all_success_gives_rate_one(self):
        rows = [_row(success=True) for _ in range(3)]
        sums = compute_arm_summaries(rows)
        s = sums[("T1", "retrieved")]
        assert s.n_runs == 3
        assert s.n_success == 3
        assert s.success_rate == pytest.approx(1.0)

    def test_no_success_gives_rate_zero(self):
        rows = [_row(success=False) for _ in range(3)]
        sums = compute_arm_summaries(rows)
        s = sums[("T1", "retrieved")]
        assert s.success_rate == pytest.approx(0.0)

    def test_partial_success_rate(self):
        rows = [
            _row(success=True),
            _row(success=True),
            _row(success=False),
        ]
        sums = compute_arm_summaries(rows)
        s = sums[("T1", "retrieved")]
        assert s.success_rate == pytest.approx(2 / 3)

    def test_medians_only_from_successes(self):
        """Failed runs must not contribute to medians."""
        rows = [
            _row(success=True,  exploratory_reads_before_edit=4,
                 total_tool_calls=10, failed_test_cycles=1),
            _row(success=True,  exploratory_reads_before_edit=6,
                 total_tool_calls=20, failed_test_cycles=3),
            _row(success=False, exploratory_reads_before_edit=99,
                 total_tool_calls=999, failed_test_cycles=99),  # must be excluded
        ]
        sums = compute_arm_summaries(rows)
        s = sums[("T1", "retrieved")]
        assert s.med_exploratory_reads == pytest.approx(5.0)   # median(4, 6)
        assert s.med_total_tool_calls == pytest.approx(15.0)   # median(10, 20)
        assert s.med_failed_test_cycles == pytest.approx(2.0)  # median(1, 3)

    def test_no_successful_runs_yields_all_none_medians(self):
        rows = [
            _row(success=False, exploratory_reads_before_edit=5,
                 total_tool_calls=10, failed_test_cycles=2,
                 elapsed_seconds=30.0, input_tokens=100, output_tokens=200),
        ]
        sums = compute_arm_summaries(rows)
        s = sums[("T1", "retrieved")]
        assert s.med_exploratory_reads is None
        assert s.med_total_tool_calls is None
        assert s.med_failed_test_cycles is None
        assert s.med_elapsed_seconds is None
        assert s.med_tokens is None

    def test_combined_token_median(self):
        rows = [
            _row(success=True, input_tokens=100, output_tokens=200),  # total=300
            _row(success=True, input_tokens=200, output_tokens=400),  # total=600
            _row(success=True, input_tokens=300, output_tokens=300),  # total=600
        ]
        sums = compute_arm_summaries(rows)
        s = sums[("T1", "retrieved")]
        # median(300, 600, 600) = 600
        assert s.med_tokens == pytest.approx(600.0)

    def test_token_median_with_none_fields(self):
        """Rows where both token fields are None should be excluded from median."""
        rows = [
            _row(success=True, input_tokens=None, output_tokens=None),  # excluded
            _row(success=True, input_tokens=100, output_tokens=200),    # total=300
        ]
        sums = compute_arm_summaries(rows)
        s = sums[("T1", "retrieved")]
        assert s.med_tokens == pytest.approx(300.0)

    def test_multiple_tasks_and_arms_are_grouped_separately(self):
        rows = [
            _row(task_id="T1", arm="retrieved", success=True),
            _row(task_id="T1", arm="empty", success=False),
            _row(task_id="T2", arm="retrieved", success=True),
            _row(task_id="T2", arm="retrieved", success=True),
        ]
        sums = compute_arm_summaries(rows)
        assert sums[("T1", "retrieved")].success_rate == pytest.approx(1.0)
        assert sums[("T1", "empty")].success_rate == pytest.approx(0.0)
        assert sums[("T2", "retrieved")].success_rate == pytest.approx(1.0)
        assert sums[("T2", "retrieved")].n_runs == 2

    def test_string_true_success_is_coerced(self):
        """SQL may return success as the string "true"."""
        rows = [_row(success="true"), _row(success="false")]  # type: ignore
        sums = compute_arm_summaries(rows)
        s = sums[("T1", "retrieved")]
        assert s.n_success == 1


# ---------------------------------------------------------------------------
# compute_paired_deltas
# ---------------------------------------------------------------------------

class TestComputePairedDeltas:

    def _make_summaries(self) -> dict[tuple[str, str], ArmSummary]:
        """Build a minimal summaries dict for T1 with all four arms."""
        def _s(arm, rate, expl, tc, fc):
            return ArmSummary(
                task_id="T1", arm=arm,
                n_runs=3, n_success=round(3 * rate),
                success_rate=rate,
                med_exploratory_reads=expl,
                med_total_tool_calls=tc,
                med_failed_test_cycles=fc,
                med_elapsed_seconds=60.0,
                med_tokens=1000.0,
            )
        return {
            ("T1", "retrieved"):     _s("retrieved",     0.67, 3.0,  8.0,  1.0),
            ("T1", "empty"):         _s("empty",          0.33, 7.0, 15.0,  2.0),
            ("T1", "static_generic"):_s("static_generic", 0.33, 5.0, 12.0,  2.0),
            ("T1", "placebo"):       _s("placebo",        0.33, 5.0, 10.0,  2.0),
        }

    def test_success_rate_delta_vs_empty(self):
        sums = self._make_summaries()
        deltas = compute_paired_deltas(sums, ["T1"])
        d = next(d for d in deltas if d.comparator_arm == "empty")
        # 0.67 - 0.33 = 0.34
        assert d.success_rate_delta == pytest.approx(0.67 - 0.33, abs=1e-9)

    def test_efficiency_delta_negative_means_retrieved_leaner(self):
        sums = self._make_summaries()
        deltas = compute_paired_deltas(sums, ["T1"])
        d = next(d for d in deltas if d.comparator_arm == "empty")
        # retrieved expl=3.0 - empty expl=7.0 = -4.0 (retrieved better)
        assert d.med_delta_exploratory_reads == pytest.approx(-4.0)

    def test_all_three_comparators_present(self):
        sums = self._make_summaries()
        deltas = compute_paired_deltas(sums, ["T1"])
        comparators = {d.comparator_arm for d in deltas}
        assert comparators == {"empty", "static_generic", "placebo"}

    def test_missing_arm_yields_none_deltas(self):
        """If retrieved arm is absent, all deltas for that task are None."""
        sums = {
            ("T1", "empty"): ArmSummary(
                task_id="T1", arm="empty",
                n_runs=3, n_success=1, success_rate=0.33,
                med_exploratory_reads=5.0, med_total_tool_calls=10.0,
                med_failed_test_cycles=1.0, med_elapsed_seconds=30.0,
                med_tokens=500.0,
            )
        }
        deltas = compute_paired_deltas(sums, ["T1"])
        d = next(d for d in deltas if d.comparator_arm == "empty")
        assert d.success_rate_delta is None
        assert d.med_delta_exploratory_reads is None

    def test_elapsed_seconds_delta_negative_means_retrieved_faster(self):
        """med_delta_elapsed_seconds < 0 means retrieved was faster."""
        # retrieved: 60s, empty: 90s  →  delta = 60 - 90 = -30 (retrieved faster)
        sums = self._make_summaries()
        # Override elapsed_seconds for retrieved and empty
        sums[("T1", "retrieved")] = ArmSummary(
            task_id="T1", arm="retrieved",
            n_runs=3, n_success=2, success_rate=0.67,
            med_exploratory_reads=3.0, med_total_tool_calls=8.0,
            med_failed_test_cycles=1.0, med_elapsed_seconds=60.0,
            med_tokens=800.0,
        )
        sums[("T1", "empty")] = ArmSummary(
            task_id="T1", arm="empty",
            n_runs=3, n_success=1, success_rate=0.33,
            med_exploratory_reads=7.0, med_total_tool_calls=15.0,
            med_failed_test_cycles=2.0, med_elapsed_seconds=90.0,
            med_tokens=1200.0,
        )
        deltas = compute_paired_deltas(sums, ["T1"])
        d = next(d for d in deltas if d.comparator_arm == "empty")
        assert d.med_delta_elapsed_seconds == pytest.approx(60.0 - 90.0)  # -30.0

    def test_elapsed_seconds_delta_none_when_no_successes(self):
        """If comparator has no successful runs, med_delta_elapsed_seconds is None."""
        sums = {
            ("T1", "retrieved"): ArmSummary(
                task_id="T1", arm="retrieved",
                n_runs=3, n_success=3, success_rate=1.0,
                med_exploratory_reads=3.0, med_total_tool_calls=8.0,
                med_failed_test_cycles=1.0, med_elapsed_seconds=60.0,
                med_tokens=800.0,
            ),
            ("T1", "empty"): ArmSummary(
                task_id="T1", arm="empty",
                n_runs=3, n_success=0, success_rate=0.0,
                med_exploratory_reads=None, med_total_tool_calls=None,
                med_failed_test_cycles=None, med_elapsed_seconds=None,
                med_tokens=None,
            ),
            ("T1", "static_generic"): ArmSummary(
                task_id="T1", arm="static_generic",
                n_runs=3, n_success=0, success_rate=0.0,
                med_exploratory_reads=None, med_total_tool_calls=None,
                med_failed_test_cycles=None, med_elapsed_seconds=None,
                med_tokens=None,
            ),
            ("T1", "placebo"): ArmSummary(
                task_id="T1", arm="placebo",
                n_runs=3, n_success=0, success_rate=0.0,
                med_exploratory_reads=None, med_total_tool_calls=None,
                med_failed_test_cycles=None, med_elapsed_seconds=None,
                med_tokens=None,
            ),
        }
        deltas = compute_paired_deltas(sums, ["T1"])
        d = next(d for d in deltas if d.comparator_arm == "empty")
        assert d.med_delta_elapsed_seconds is None

    def test_tokens_delta_negative_means_retrieved_cheaper(self):
        """med_delta_tokens < 0 means retrieved used fewer tokens."""
        # retrieved: 800, placebo: 1500  →  delta = 800 - 1500 = -700
        sums = {
            ("T1", "retrieved"): ArmSummary(
                task_id="T1", arm="retrieved",
                n_runs=3, n_success=3, success_rate=1.0,
                med_exploratory_reads=3.0, med_total_tool_calls=8.0,
                med_failed_test_cycles=1.0, med_elapsed_seconds=50.0,
                med_tokens=800.0,
            ),
            ("T1", "empty"): ArmSummary(
                task_id="T1", arm="empty",
                n_runs=3, n_success=1, success_rate=0.33,
                med_exploratory_reads=7.0, med_total_tool_calls=15.0,
                med_failed_test_cycles=2.0, med_elapsed_seconds=80.0,
                med_tokens=1200.0,
            ),
            ("T1", "static_generic"): ArmSummary(
                task_id="T1", arm="static_generic",
                n_runs=3, n_success=1, success_rate=0.33,
                med_exploratory_reads=6.0, med_total_tool_calls=12.0,
                med_failed_test_cycles=2.0, med_elapsed_seconds=75.0,
                med_tokens=1100.0,
            ),
            ("T1", "placebo"): ArmSummary(
                task_id="T1", arm="placebo",
                n_runs=3, n_success=1, success_rate=0.33,
                med_exploratory_reads=5.0, med_total_tool_calls=10.0,
                med_failed_test_cycles=2.0, med_elapsed_seconds=70.0,
                med_tokens=1500.0,
            ),
        }
        deltas = compute_paired_deltas(sums, ["T1"])
        d = next(d for d in deltas if d.comparator_arm == "placebo")
        assert d.med_delta_tokens == pytest.approx(800.0 - 1500.0)  # -700.0

    def test_tokens_delta_none_when_retrieved_has_no_successes(self):
        """If retrieved has no successful runs, med_delta_tokens is None."""
        sums = {
            ("T1", "retrieved"): ArmSummary(
                task_id="T1", arm="retrieved",
                n_runs=3, n_success=0, success_rate=0.0,
                med_exploratory_reads=None, med_total_tool_calls=None,
                med_failed_test_cycles=None, med_elapsed_seconds=None,
                med_tokens=None,  # no successes → None
            ),
            ("T1", "empty"): ArmSummary(
                task_id="T1", arm="empty",
                n_runs=3, n_success=2, success_rate=0.67,
                med_exploratory_reads=6.0, med_total_tool_calls=12.0,
                med_failed_test_cycles=2.0, med_elapsed_seconds=70.0,
                med_tokens=1200.0,
            ),
            ("T1", "static_generic"): ArmSummary(
                task_id="T1", arm="static_generic",
                n_runs=3, n_success=0, success_rate=0.0,
                med_exploratory_reads=None, med_total_tool_calls=None,
                med_failed_test_cycles=None, med_elapsed_seconds=None,
                med_tokens=None,
            ),
            ("T1", "placebo"): ArmSummary(
                task_id="T1", arm="placebo",
                n_runs=3, n_success=0, success_rate=0.0,
                med_exploratory_reads=None, med_total_tool_calls=None,
                med_failed_test_cycles=None, med_elapsed_seconds=None,
                med_tokens=None,
            ),
        }
        deltas = compute_paired_deltas(sums, ["T1"])
        d = next(d for d in deltas if d.comparator_arm == "empty")
        assert d.med_delta_tokens is None


# ---------------------------------------------------------------------------
# compute_falsifiable_checks
# ---------------------------------------------------------------------------

class TestComputeFalsifiableChecks:

    def _make_sums(
        self,
        r_rate: float,
        p_rate: float,
        r_expl: Optional[float] = None,
        p_expl: Optional[float] = None,
        r_tc: Optional[float] = None,
        p_tc: Optional[float] = None,
        r_fc: Optional[float] = None,
        p_fc: Optional[float] = None,
    ) -> dict[tuple[str, str], ArmSummary]:
        def _s(arm, rate, expl, tc, fc):
            return ArmSummary(
                task_id="T1", arm=arm,
                n_runs=3, n_success=round(3 * rate),
                success_rate=rate,
                med_exploratory_reads=expl,
                med_total_tool_calls=tc,
                med_failed_test_cycles=fc,
                med_elapsed_seconds=None,
                med_tokens=None,
            )
        return {
            ("T1", "retrieved"): _s("retrieved", r_rate, r_expl, r_tc, r_fc),
            ("T1", "placebo"):   _s("placebo",   p_rate, p_expl, p_tc, p_fc),
            ("T1", "empty"):     _s("empty",     0.0,    None,   None, None),
            ("T1", "static_generic"): _s("static_generic", 0.0, None, None, None),
        }

    # (a) retrieved clearly > placebo → PASS
    def test_retrieved_beats_placebo_on_rate(self):
        sums = self._make_sums(r_rate=1.0, p_rate=0.33)
        checks = compute_falsifiable_checks(sums, ["T1"])
        c = checks[0]
        assert c.verdict == "PASS"
        assert c.basis == "success-rate"

    # (b) retrieved < placebo → FLAG
    def test_retrieved_below_placebo_on_rate(self):
        sums = self._make_sums(r_rate=0.0, p_rate=0.67)
        checks = compute_falsifiable_checks(sums, ["T1"])
        c = checks[0]
        assert c.verdict == "FLAG"
        assert c.basis == "success-rate"

    # (c) rate tie, retrieved has fewer exploratory reads → PASS
    def test_rate_tie_broken_by_exploratory_reads_retrieved_wins(self):
        sums = self._make_sums(
            r_rate=0.67, p_rate=0.67,
            r_expl=3.0, p_expl=7.0,  # retrieved leaner
        )
        checks = compute_falsifiable_checks(sums, ["T1"])
        c = checks[0]
        assert c.verdict == "PASS"
        assert "exploratory_reads_before_edit" in c.basis
        assert "tiebreak" in c.basis.lower()

    # (d) rate tie, retrieved has more reads → FLAG
    def test_rate_tie_broken_by_exploratory_reads_retrieved_loses(self):
        sums = self._make_sums(
            r_rate=0.67, p_rate=0.67,
            r_expl=9.0, p_expl=4.0,  # retrieved heavier
        )
        checks = compute_falsifiable_checks(sums, ["T1"])
        c = checks[0]
        assert c.verdict == "FLAG"
        assert "exploratory_reads_before_edit" in c.basis

    # (e) first tiebreak metric equal; second decides (tool_calls)
    def test_rate_tie_first_metric_equal_second_decides(self):
        sums = self._make_sums(
            r_rate=0.67, p_rate=0.67,
            r_expl=5.0, p_expl=5.0,  # first metric tied
            r_tc=8.0,   p_tc=12.0,   # second metric: retrieved wins
        )
        checks = compute_falsifiable_checks(sums, ["T1"])
        c = checks[0]
        assert c.verdict == "PASS"
        assert "total_tool_calls" in c.basis

    # (e-alt) first two equal, third decides
    def test_rate_tie_first_two_equal_third_decides(self):
        sums = self._make_sums(
            r_rate=0.67, p_rate=0.67,
            r_expl=5.0, p_expl=5.0,
            r_tc=8.0,   p_tc=8.0,
            r_fc=1.0,   p_fc=3.0,   # third metric: retrieved wins
        )
        checks = compute_falsifiable_checks(sums, ["T1"])
        c = checks[0]
        assert c.verdict == "PASS"
        assert "failed_test_cycles" in c.basis

    # (f) all tiebreak metrics equal → FLAG
    def test_all_tiebreak_metrics_equal_yields_flag(self):
        sums = self._make_sums(
            r_rate=0.67, p_rate=0.67,
            r_expl=5.0, p_expl=5.0,
            r_tc=10.0,  p_tc=10.0,
            r_fc=2.0,   p_fc=2.0,
        )
        checks = compute_falsifiable_checks(sums, ["T1"])
        c = checks[0]
        assert c.verdict == "FLAG"
        assert "all-tied" in c.basis

    # (g) missing arm data
    def test_missing_retrieved_arm_yields_flag_missing_data(self):
        sums = {
            ("T1", "placebo"): ArmSummary(
                task_id="T1", arm="placebo",
                n_runs=3, n_success=1, success_rate=0.33,
                med_exploratory_reads=5.0, med_total_tool_calls=10.0,
                med_failed_test_cycles=1.0, med_elapsed_seconds=30.0,
                med_tokens=500.0,
            ),
        }
        checks = compute_falsifiable_checks(sums, ["T1"])
        c = checks[0]
        assert c.verdict == "FLAG"
        assert c.basis == "missing-data"

    def test_missing_placebo_arm_yields_flag_missing_data(self):
        sums = {
            ("T1", "retrieved"): ArmSummary(
                task_id="T1", arm="retrieved",
                n_runs=3, n_success=3, success_rate=1.0,
                med_exploratory_reads=3.0, med_total_tool_calls=8.0,
                med_failed_test_cycles=1.0, med_elapsed_seconds=20.0,
                med_tokens=400.0,
            ),
        }
        checks = compute_falsifiable_checks(sums, ["T1"])
        c = checks[0]
        assert c.verdict == "FLAG"
        assert c.basis == "missing-data"


# ---------------------------------------------------------------------------
# compute_report (top-level, integration)
# ---------------------------------------------------------------------------

class TestComputeReport:

    def test_empty_input_returns_empty_report(self):
        report = compute_report([])
        assert report.total_rows == 0
        assert report.task_ids == []
        assert report.arm_summaries == []
        assert report.paired_deltas == []
        assert report.falsifiable_checks == []

    def test_preserves_task_id_insertion_order(self):
        rows = [
            _row(task_id="T2", arm="retrieved", success=True),
            _row(task_id="T1", arm="retrieved", success=True),
            _row(task_id="T3", arm="retrieved", success=True),
        ]
        report = compute_report(rows)
        assert report.task_ids == ["T2", "T1", "T3"]

    def test_total_rows_count(self):
        rows = [_row() for _ in range(7)]
        report = compute_report(rows)
        assert report.total_rows == 7

    def test_falsifiable_check_produced_per_task(self):
        rows = [
            _row(task_id="T1", arm="retrieved", success=True),
            _row(task_id="T1", arm="placebo",   success=False),
            _row(task_id="T2", arm="retrieved", success=True),
            _row(task_id="T2", arm="placebo",   success=False),
        ]
        report = compute_report(rows)
        check_tasks = {c.task_id for c in report.falsifiable_checks}
        assert check_tasks == {"T1", "T2"}

    def test_full_matrix_three_tasks_four_arms(self):
        """Simulate the target N=3 repeats × 3 tasks × 4 arms scenario."""
        arms = ["empty", "static_generic", "retrieved", "placebo"]
        rows = []
        for tid in ["T1", "T2", "T3"]:
            for arm in arms:
                for rep in range(1, 4):
                    # retrieved always succeeds; others sometimes do
                    success = arm == "retrieved" or (arm == "placebo" and rep == 1)
                    rows.append(_row(
                        task_id=tid, arm=arm, repeat_number=rep,
                        success=success,
                        exploratory_reads_before_edit=3 if arm == "retrieved" else 8,
                        total_tool_calls=10 if arm == "retrieved" else 20,
                        failed_test_cycles=1 if success else 3,
                        elapsed_seconds=60.0,
                        input_tokens=500, output_tokens=300,
                    ))
        report = compute_report(rows)
        assert report.task_ids == ["T1", "T2", "T3"]
        # retrieved > placebo on success rate for all tasks → PASS
        for c in report.falsifiable_checks:
            assert c.verdict == "PASS", f"{c.task_id}: {c}"
            assert c.basis == "success-rate"

    def test_paired_deltas_count(self):
        """Each task should have exactly 3 paired deltas (one per comparator)."""
        rows = [
            _row(task_id="T1", arm=arm, success=True)
            for arm in ["empty", "static_generic", "retrieved", "placebo"]
        ]
        report = compute_report(rows)
        t1_deltas = [d for d in report.paired_deltas if d.task_id == "T1"]
        assert len(t1_deltas) == 3
        comparators = {d.comparator_arm for d in t1_deltas}
        assert comparators == {"empty", "static_generic", "placebo"}


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------

class TestRenderReport:

    def test_empty_report_renders_no_runs_message(self):
        md = render_report(compute_report([]))
        assert "no runs yet" in md.lower() or "empty" in md.lower()
        # Must not crash or produce section headers
        assert "## 1." not in md

    def test_non_empty_report_has_three_sections(self):
        rows = [
            _row(task_id="T1", arm="retrieved", success=True,
                 exploratory_reads_before_edit=3, total_tool_calls=10,
                 failed_test_cycles=1, elapsed_seconds=30.0,
                 input_tokens=500, output_tokens=300),
            _row(task_id="T1", arm="placebo", success=False),
        ]
        md = render_report(compute_report(rows))
        assert "## 1." in md
        assert "## 2." in md
        assert "## 3." in md

    def test_directional_label_present_in_report(self):
        rows = [_row(task_id="T1", arm="retrieved", success=True)]
        md = render_report(compute_report(rows))
        assert "directional" in md

    def test_n_equals_label_present(self):
        rows = [_row(task_id="T1", arm="retrieved", success=True)]
        md = render_report(compute_report(rows))
        assert "N=3" in md

    def test_pass_verdict_appears_in_rendered_table(self):
        rows = [
            _row(task_id="T1", arm="retrieved", success=True),
            _row(task_id="T1", arm="retrieved", success=True),
            _row(task_id="T1", arm="retrieved", success=True),
            _row(task_id="T1", arm="placebo",   success=False),
            _row(task_id="T1", arm="placebo",   success=False),
            _row(task_id="T1", arm="placebo",   success=False),
        ]
        md = render_report(compute_report(rows))
        assert "PASS" in md

    def test_flag_verdict_appears_in_rendered_table(self):
        rows = [
            _row(task_id="T1", arm="retrieved", success=False),
            _row(task_id="T1", arm="placebo",   success=True),
        ]
        md = render_report(compute_report(rows))
        assert "FLAG" in md

    def test_em_dash_for_none_medians(self):
        """Cells with no data should render as '—', not 'None'."""
        rows = [_row(task_id="T1", arm="retrieved", success=False)]
        md = render_report(compute_report(rows))
        assert "None" not in md
        assert "—" in md

    def test_tiebreak_basis_is_documented_in_table(self):
        rows = [
            _row(task_id="T1", arm="retrieved", success=True,
                 exploratory_reads_before_edit=3),
            _row(task_id="T1", arm="placebo",   success=True,
                 exploratory_reads_before_edit=7),
        ]
        md = render_report(compute_report(rows))
        # The basis column must say "tiebreak" so reader knows it was a tie
        assert "tiebreak" in md.lower()
