"""Tests for src/harness/metrics.py — Task 5.2.

Synthetic trace schema:
  trace: list[dict], each event {"type": str, "target": str}
  type in {"read", "search", "edit", "test"}
  test events additionally carry {"passed": bool}
"""
import pytest
from harness.metrics import exploratory_reads_before_edit, failed_test_cycles


# ---------------------------------------------------------------------------
# exploratory_reads_before_edit
# ---------------------------------------------------------------------------

class TestExploratoryReadsBeforeEdit:

    def test_empty_trace_returns_zero(self):
        assert exploratory_reads_before_edit([], ["foo.py"]) == 0

    def test_reads_and_searches_before_first_required_edit_are_counted(self):
        trace = [
            {"type": "read",   "target": "a.py"},
            {"type": "search", "target": "some query"},
            {"type": "read",   "target": "b.py"},
            {"type": "edit",   "target": "target.py"},   # first edit to required file
            {"type": "read",   "target": "c.py"},         # after — not counted
        ]
        # 3 exploratory events before the first required edit
        assert exploratory_reads_before_edit(trace, ["target.py"]) == 3

    def test_reads_after_required_edit_are_not_counted(self):
        trace = [
            {"type": "read",   "target": "a.py"},
            {"type": "edit",   "target": "target.py"},
            {"type": "read",   "target": "b.py"},
            {"type": "search", "target": "more searching"},
        ]
        assert exploratory_reads_before_edit(trace, ["target.py"]) == 1

    def test_edit_to_non_required_file_does_not_stop_count(self):
        trace = [
            {"type": "read",   "target": "a.py"},
            {"type": "edit",   "target": "other.py"},    # NOT a required file — does not stop
            {"type": "search", "target": "query"},
            {"type": "read",   "target": "b.py"},
            {"type": "edit",   "target": "target.py"},   # first required edit — stops here
            {"type": "read",   "target": "c.py"},
        ]
        # 3 exploratory events (read a.py, search query, read b.py) before required edit
        assert exploratory_reads_before_edit(trace, ["target.py"]) == 3

    def test_no_edit_to_required_file_counts_all_reads_searches(self):
        trace = [
            {"type": "read",   "target": "a.py"},
            {"type": "search", "target": "query"},
            {"type": "edit",   "target": "unrelated.py"},
            {"type": "read",   "target": "b.py"},
        ]
        # Agent never reached a required file — all 3 read/search events count
        assert exploratory_reads_before_edit(trace, ["target.py"]) == 3

    def test_multiple_required_files_stops_at_first_matching_edit(self):
        trace = [
            {"type": "read",   "target": "a.py"},
            {"type": "edit",   "target": "req2.py"},     # first required file hit
            {"type": "search", "target": "more"},
            {"type": "edit",   "target": "req1.py"},
        ]
        # req2.py is in required_files, so count stops after 1 read
        assert exploratory_reads_before_edit(trace, ["req1.py", "req2.py"]) == 1

    def test_zero_reads_before_immediate_required_edit(self):
        trace = [
            {"type": "edit", "target": "target.py"},
            {"type": "read", "target": "a.py"},
        ]
        assert exploratory_reads_before_edit(trace, ["target.py"]) == 0

    def test_only_test_and_edit_events_no_reads_searches(self):
        trace = [
            {"type": "test", "target": "test_suite", "passed": True},
            {"type": "edit", "target": "target.py"},
        ]
        assert exploratory_reads_before_edit(trace, ["target.py"]) == 0

    def test_empty_required_files_counts_all_reads_searches(self):
        """No required files specified → agent can never reach one → count all."""
        trace = [
            {"type": "read",   "target": "a.py"},
            {"type": "search", "target": "query"},
        ]
        assert exploratory_reads_before_edit(trace, []) == 2


# ---------------------------------------------------------------------------
# failed_test_cycles
# ---------------------------------------------------------------------------

class TestFailedTestCycles:

    def test_empty_trace_returns_zero(self):
        assert failed_test_cycles([]) == 0

    def test_no_test_events_returns_zero(self):
        trace = [
            {"type": "read",   "target": "a.py"},
            {"type": "edit",   "target": "b.py"},
        ]
        assert failed_test_cycles(trace) == 0

    def test_all_passing_tests_returns_zero(self):
        trace = [
            {"type": "test", "target": "suite", "passed": True},
            {"type": "test", "target": "suite", "passed": True},
        ]
        assert failed_test_cycles(trace) == 0

    def test_all_failing_tests_counted(self):
        trace = [
            {"type": "test", "target": "suite", "passed": False},
            {"type": "test", "target": "suite", "passed": False},
            {"type": "test", "target": "suite", "passed": False},
        ]
        assert failed_test_cycles(trace) == 3

    def test_mixed_pass_fail_counts_only_failures(self):
        trace = [
            {"type": "test", "target": "suite", "passed": False},
            {"type": "test", "target": "suite", "passed": True},
            {"type": "test", "target": "suite", "passed": False},
            {"type": "test", "target": "suite", "passed": True},
            {"type": "test", "target": "suite", "passed": False},
        ]
        assert failed_test_cycles(trace) == 3

    def test_non_test_events_ignored(self):
        trace = [
            {"type": "read",   "target": "a.py"},
            {"type": "test",   "target": "suite", "passed": False},
            {"type": "edit",   "target": "b.py"},
            {"type": "test",   "target": "suite", "passed": True},
            {"type": "search", "target": "query"},
        ]
        assert failed_test_cycles(trace) == 1
