"""
Tests for src/harness/run.py — Task 5.1.

All tests are offline: no real agent subprocess, no UC calls, no Volume uploads.

Fixtures:
  - fake_git_repo: a minimal real git repo in tmp_path (used as platform_repo);
    provides a starting_commit the git worktree machinery can use.
  - canned_artifact: a dict matching the memory_artifacts schema.
  - canned_task_def: a minimal task definition that uses a fast acceptance command.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import patch

import pytest

from harness.run import (
    T1_ORACLE_COMMIT,
    T1_ORACLE_FIXTURE_PATH,
    AgentRunResult,
    EvalRunRow,
    _build_insert_sql,
    _capture_working_tree_state,
    _check_forbidden_unchanged,
    _extract_sentinel_observed,
    _parse_agent_echo,
    _parse_usage,
    adapt_stream_json_to_trace,
    run_matrix,
    run_one,
)


def _noop_oracle(repo: str, commit: str, path: str) -> str:
    """Stub oracle for tests that don't test oracle behaviour.

    Returns minimal non-empty content so run_one can write the file
    without hitting the real git repo.
    """
    return f"# stub oracle for {path}\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_agent_result(
    *,
    sentinel: str = "mem-T1-retrieved-abc123",
    total_tool_calls: int = 5,
    stopped_reason: Optional[str] = "agent_exit",
    trace: Optional[list[dict]] = None,
    raw_events: Optional[list[dict]] = None,
    tokens_in: int = 100,
    tokens_out: int = 50,
    elapsed: float = 10.0,
    final_commit: Optional[str] = "deadbeef",
) -> AgentRunResult:
    return AgentRunResult(
        trace=trace or [],
        agent_echo=f"I read the memory. {sentinel} is acknowledged.",
        total_tool_calls=total_tool_calls,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        elapsed_seconds=elapsed,
        final_commit=final_commit,
        exit_code=0,
        stopped_reason=stopped_reason,
        raw_events=raw_events or [],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_git_repo(tmp_path):
    """Create a minimal real git repo to use as platform_repo.

    Returns (repo_path: str, starting_commit: str).
    """
    repo = tmp_path / "platform"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@t.com",
           "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@t.com"}

    def git(*args):
        subprocess.run(["git"] + list(args), cwd=str(repo), check=True,
                       capture_output=True, env=env)

    git("init")
    git("config", "user.email", "test@test.com")
    git("config", "user.name", "Test")
    (repo / "README.md").write_text("fake platform repo\n")
    git("add", ".")
    git("commit", "-m", "init")

    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo), capture_output=True, text=True,
    )
    commit = r.stdout.strip()
    return str(repo), commit


@pytest.fixture
def canned_memory_markdown():
    return "<!-- MEMORY_SENTINEL: mem-T1-retrieved-abc123 -->\n# MEMORY.md\nfoo bar\n"


@pytest.fixture
def canned_artifact(canned_memory_markdown):
    md_bytes = canned_memory_markdown.encode("utf-8")
    sha = _sha256(md_bytes)
    return {
        "artifact_id": "art-001",
        "task_id": "T1",
        "arm": "retrieved",
        "file_sha256": sha,
        "sentinel": "mem-T1-retrieved-abc123",
        "memory_markdown": canned_memory_markdown,
        "volume_path": "/Volumes/cat/sch/vol/artifacts/T1/retrieved/MEMORY.md",
    }


@pytest.fixture
def canned_task_def(fake_git_repo):
    """A minimal task definition using a fast acceptance command.

    Uses task_id="T_TEST" (not "T1") so oracle logic is not triggered —
    tests that specifically test oracle behaviour construct their own T1 task_def.
    """
    _, starting_commit = fake_git_repo
    return {
        "task_id": "T_TEST",
        "goal_prompt": "/goal do something",
        "repository": "~/Projects/platform",
        "starting_commit": starting_commit,
        "acceptance_command": f"{sys.executable} -c 'import sys; sys.exit(0)'",
        "regression_command": None,
        "required_files": ["src/main.py"],
        "forbidden_files": ["forbidden.py"],
        "max_minutes": 30,
        "max_tool_calls": 120,
    }


# ---------------------------------------------------------------------------
# Tests: stream-json → trace adapter
# ---------------------------------------------------------------------------

class TestAdaptStreamJsonToTrace:
    """Tests for adapt_stream_json_to_trace (pure, no I/O)."""

    def _make_tool_use_event(self, tool_id, name, input_dict):
        return {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": tool_id, "name": name, "input": input_dict}
                ],
            },
        }

    def _make_tool_result_event(self, tool_id, content="", is_error=False):
        return {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": content,
                        "is_error": is_error,
                    }
                ],
            },
        }

    def test_empty_events_returns_empty_trace(self):
        assert adapt_stream_json_to_trace([]) == []

    def test_read_tool_maps_to_read(self):
        events = [self._make_tool_use_event("t1", "Read", {"file_path": "src/foo.py"})]
        trace = adapt_stream_json_to_trace(events)
        assert trace == [{"type": "read", "target": "src/foo.py"}]

    def test_glob_tool_maps_to_search(self):
        events = [self._make_tool_use_event("t1", "Glob", {"pattern": "**/*.py"})]
        trace = adapt_stream_json_to_trace(events)
        assert trace == [{"type": "search", "target": "**/*.py"}]

    def test_grep_tool_maps_to_search(self):
        events = [self._make_tool_use_event("t1", "Grep", {"pattern": "def auto_k"})]
        trace = adapt_stream_json_to_trace(events)
        assert trace == [{"type": "search", "target": "def auto_k"}]

    def test_edit_tool_maps_to_edit(self):
        events = [
            self._make_tool_use_event(
                "t1", "Edit",
                {"file_path": "src/foo.py", "old_string": "x", "new_string": "y"}
            )
        ]
        trace = adapt_stream_json_to_trace(events)
        assert trace == [{"type": "edit", "target": "src/foo.py"}]

    def test_write_tool_maps_to_edit(self):
        events = [
            self._make_tool_use_event("t1", "Write", {"file_path": "src/new.py"})
        ]
        trace = adapt_stream_json_to_trace(events)
        assert trace == [{"type": "edit", "target": "src/new.py"}]

    def test_bash_pytest_maps_to_test_passed(self):
        events = [
            self._make_tool_use_event("t1", "Bash", {"command": "pytest tests/ -q"}),
            self._make_tool_result_event("t1", content="1 passed", is_error=False),
        ]
        trace = adapt_stream_json_to_trace(events)
        assert trace == [{"type": "test", "target": "pytest tests/ -q", "passed": True}]

    def test_bash_pytest_failed_maps_to_test_not_passed(self):
        events = [
            self._make_tool_use_event("t1", "Bash", {"command": "pytest tests/ -q"}),
            self._make_tool_result_event("t1", content="1 failed", is_error=True),
        ]
        trace = adapt_stream_json_to_trace(events)
        assert trace == [{"type": "test", "target": "pytest tests/ -q", "passed": False}]

    def test_bash_non_test_command_omitted(self):
        events = [
            self._make_tool_use_event("t1", "Bash", {"command": "ls -la"}),
        ]
        trace = adapt_stream_json_to_trace(events)
        assert trace == []

    def test_unknown_tool_omitted(self):
        events = [
            self._make_tool_use_event("t1", "UnknownTool", {"input": "stuff"}),
        ]
        trace = adapt_stream_json_to_trace(events)
        assert trace == []

    def test_mixed_sequence_preserves_order(self):
        events = [
            self._make_tool_use_event("t1", "Read", {"file_path": "a.py"}),
            self._make_tool_use_event("t2", "Edit", {"file_path": "b.py"}),
            self._make_tool_use_event("t3", "Bash", {"command": "pytest x.py"}),
            self._make_tool_result_event("t3", is_error=False),
            self._make_tool_use_event("t4", "Glob", {"pattern": "*.py"}),
        ]
        trace = adapt_stream_json_to_trace(events)
        assert [e["type"] for e in trace] == ["read", "edit", "test", "search"]

    def test_ruff_command_maps_to_test(self):
        events = [
            self._make_tool_use_event("t1", "Bash", {"command": "ruff check src/"}),
            self._make_tool_result_event("t1", is_error=False),
        ]
        trace = adapt_stream_json_to_trace(events)
        assert trace == [{"type": "test", "target": "ruff check src/", "passed": True}]

    def test_multiple_tool_uses_in_one_event(self):
        event = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read",
                     "input": {"file_path": "a.py"}},
                    {"type": "tool_use", "id": "t2", "name": "Read",
                     "input": {"file_path": "b.py"}},
                ],
            },
        }
        trace = adapt_stream_json_to_trace([event])
        assert len(trace) == 2
        assert trace[0] == {"type": "read", "target": "a.py"}
        assert trace[1] == {"type": "read", "target": "b.py"}


# ---------------------------------------------------------------------------
# Tests: _parse_agent_echo
# ---------------------------------------------------------------------------

class TestParseAgentEcho:

    def test_extracts_text_from_assistant_messages(self):
        events = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello sentinel mem-abc"}],
                },
            }
        ]
        echo = _parse_agent_echo(events)
        assert "Hello sentinel mem-abc" in echo

    def test_empty_events_returns_empty_string(self):
        assert _parse_agent_echo([]) == ""

    def test_non_assistant_events_ignored(self):
        events = [{"type": "result", "text": "should not appear"}]
        assert _parse_agent_echo(events) == ""

    def test_multiple_text_blocks_joined(self):
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Part one."},
                        {"type": "text", "text": "Part two."},
                    ]
                },
            }
        ]
        echo = _parse_agent_echo(events)
        assert "Part one." in echo
        assert "Part two." in echo


# ---------------------------------------------------------------------------
# Tests: _parse_usage
# ---------------------------------------------------------------------------

class TestParseUsage:

    def test_extracts_from_result_event(self):
        events = [
            {
                "type": "result",
                "subtype": "success",
                "usage": {"input_tokens": 1200, "output_tokens": 350},
            }
        ]
        assert _parse_usage(events) == (1200, 350)

    def test_returns_zeros_when_no_usage(self):
        assert _parse_usage([]) == (0, 0)

    def test_fallback_to_any_event_with_usage(self):
        events = [
            {"type": "assistant", "usage": {"input_tokens": 100, "output_tokens": 30}},
        ]
        assert _parse_usage(events) == (100, 30)


# ---------------------------------------------------------------------------
# Tests: _build_insert_sql (pure)
# ---------------------------------------------------------------------------

class TestBuildInsertSql:

    def _make_row(self, **overrides) -> EvalRunRow:
        now = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)
        defaults = dict(
            run_id="run-123",
            task_id="T1",
            arm="retrieved",
            repeat_number=1,
            artifact_id="art-001",
            file_sha256="a" * 64,
            sentinel_expected="mem-T1-retrieved-abc",
            sentinel_observed="mem-T1-retrieved-abc",
            injection_verified=True,
            agent_name="claude-code",
            model_name="databricks-claude-sonnet-4-5",
            starting_commit="ac6a62b",
            final_commit="deadbeef",
            success=True,
            acceptance_tests_passed=True,
            regression_tests_passed=None,
            forbidden_files_unchanged=True,
            elapsed_seconds=42.5,
            total_tool_calls=10,
            exploratory_reads_before_edit=3,
            failed_test_cycles=1,
            input_tokens=1000,
            output_tokens=200,
            trace_path="/Volumes/.../transcript.jsonl",
            final_diff_path="/Volumes/.../final.diff",
            failure_reason=None,
            started_at=now,
            completed_at=now,
        )
        defaults.update(overrides)
        return EvalRunRow(**defaults)

    def test_sql_contains_table_name(self):
        sql = _build_insert_sql(self._make_row())
        assert "eval_runs" in sql

    def test_sql_contains_all_columns(self):
        sql = _build_insert_sql(self._make_row())
        for col in [
            "run_id", "task_id", "arm", "repeat_number", "artifact_id",
            "file_sha256", "sentinel_expected", "sentinel_observed",
            "injection_verified", "agent_name", "model_name", "starting_commit",
            "final_commit", "success", "acceptance_tests_passed",
            "regression_tests_passed", "forbidden_files_unchanged",
            "elapsed_seconds", "total_tool_calls", "exploratory_reads_before_edit",
            "failed_test_cycles", "input_tokens", "output_tokens",
            "trace_path", "final_diff_path", "failure_reason",
            "started_at", "completed_at",
        ]:
            assert col in sql, f"Column '{col}' missing from INSERT SQL"

    def test_null_nullable_fields(self):
        row = self._make_row(
            sentinel_observed=None,
            final_commit=None,
            regression_tests_passed=None,
            failure_reason=None,
            completed_at=None,
        )
        sql = _build_insert_sql(row)
        # Count NULLs — should have 5
        assert sql.count("NULL") >= 5

    def test_boolean_true_false(self):
        sql = _build_insert_sql(self._make_row(success=True, injection_verified=False))
        assert "TRUE" in sql
        assert "FALSE" in sql

    def test_string_with_single_quote_escaped(self):
        row = self._make_row(failure_reason="agent's failure")
        sql = _build_insert_sql(row)
        assert "agent''s failure" in sql

    def test_timestamp_cast(self):
        sql = _build_insert_sql(self._make_row())
        assert "CAST(" in sql and "AS TIMESTAMP)" in sql


# ---------------------------------------------------------------------------
# Tests: _extract_sentinel_observed
# ---------------------------------------------------------------------------

class TestExtractSentinelObserved:

    def test_returns_sentinel_when_present(self):
        result = _extract_sentinel_observed("I ack mem-abc-123.", "mem-abc-123")
        assert result == "mem-abc-123"

    def test_returns_none_when_absent(self):
        result = _extract_sentinel_observed("nothing here", "mem-abc-123")
        assert result is None


# ---------------------------------------------------------------------------
# Fixtures for run_one integration tests
# ---------------------------------------------------------------------------

def _noop_write(row, wh, profile):
    """Stub write_row_fn that does nothing."""
    pass


def _noop_upload(local, vol, profile):
    """Stub upload_fn that does nothing."""
    pass


# ---------------------------------------------------------------------------
# Tests: run_one — injection-verified path
# ---------------------------------------------------------------------------

class TestRunOneInjectionVerified:

    def test_injection_verified_true_when_sentinel_echoed(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(sentinel=sentinel)

        rows_written = []

        def capture_write(row, wh, profile):
            rows_written.append(row)

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=capture_write,
            upload_fn=_noop_upload,
        )

        assert row.injection_verified is True
        assert row.sentinel_observed == sentinel
        assert len(rows_written) == 1

    def test_injection_verified_false_when_sentinel_not_echoed(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        platform_repo, _ = fake_git_repo

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            result = _make_agent_result(sentinel="WRONG_SENTINEL")
            # Override agent_echo to NOT contain the expected sentinel
            result.agent_echo = "I did stuff but forgot the sentinel."
            return result

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )

        assert row.injection_verified is False
        assert row.sentinel_observed is None


# ---------------------------------------------------------------------------
# Tests: run_one — sha-mismatch abort
# ---------------------------------------------------------------------------

class TestRunOneShaMismatch:

    def test_integrity_failure_on_sha_mismatch(
        self, fake_git_repo, canned_task_def
    ):
        platform_repo, _ = fake_git_repo

        # Artifact with a sha that won't match the staged content
        bad_artifact = {
            "artifact_id": "art-bad",
            "task_id": "T1",
            "arm": "retrieved",
            "file_sha256": "b" * 64,  # wrong sha
            "sentinel": "mem-bad",
            "memory_markdown": "# MEMORY\nsome content\n",
            "volume_path": "/Volumes/.../MEMORY.md",
        }

        agent_called = []

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            agent_called.append(True)
            return _make_agent_result()

        rows_written = []

        def capture_write(row, wh, profile):
            rows_written.append(row)

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=bad_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=capture_write,
            upload_fn=_noop_upload,
        )

        assert row.failure_reason == "integrity_failure"
        assert row.success is False
        assert len(agent_called) == 0, "Agent should not be called on sha mismatch"
        assert len(rows_written) == 1


# ---------------------------------------------------------------------------
# Tests: run_one — success combos (acceptance / regression / forbidden)
# ---------------------------------------------------------------------------

class TestRunOneSuccessCombos:

    def _run(
        self,
        fake_git_repo,
        canned_artifact,
        acceptance_exit: int = 0,
        regression_exit: Optional[int] = None,
        has_regression_cmd: bool = False,
        corrupt_forbidden: bool = False,
    ) -> EvalRunRow:
        platform_repo, starting_commit = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        # If corrupt_forbidden, commit a change to the forbidden file
        if corrupt_forbidden:
            env = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@t.com",
                   "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@t.com"}
            forbidden_path = os.path.join(platform_repo, "forbidden.py")
            # We'll simulate this by making the forbidden diff check see a change
            # by writing directly to the worktree (not committing here — the check
            # uses starting_commit..HEAD so uncommitted changes don't appear;
            # to test forbidden detection we need a committed change).
            # We'll do a real commit in the worktree via the stub_agent.
            pass  # handled below

        acc_cmd = f"{sys.executable} -c 'import sys; sys.exit({acceptance_exit})'"
        reg_cmd = None
        if has_regression_cmd:
            rc = regression_exit if regression_exit is not None else 0
            reg_cmd = f"{sys.executable} -c 'import sys; sys.exit({rc})'"

        task_def = {
            "task_id": "T_TEST",
            "goal_prompt": "/goal do something",
            "repository": "~/Projects/platform",
            "starting_commit": starting_commit,
            "acceptance_command": acc_cmd,
            "regression_command": reg_cmd,
            "required_files": ["src/main.py"],
            "forbidden_files": ["forbidden.py"],
            "max_minutes": 30,
            "max_tool_calls": 120,
        }

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            if corrupt_forbidden:
                # Commit a change to forbidden.py inside the worktree
                forbidden_wt = os.path.join(worktree, "forbidden.py")
                with open(forbidden_wt, "w") as f:
                    f.write("# corrupted\n")
                env2 = {**os.environ,
                        "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@t.com",
                        "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@t.com"}
                subprocess.run(["git", "add", "."], cwd=worktree, env=env2,
                                capture_output=True)
                subprocess.run(["git", "commit", "-m", "corrupt forbidden"],
                                cwd=worktree, env=env2, capture_output=True)
            return _make_agent_result(sentinel=sentinel)

        row = run_one(
            task_def=task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )
        return row

    def test_all_pass_means_success(self, fake_git_repo, canned_artifact):
        row = self._run(fake_git_repo, canned_artifact, acceptance_exit=0)
        assert row.success is True
        assert row.acceptance_tests_passed is True
        assert row.forbidden_files_unchanged is True

    def test_acceptance_fail_means_no_success(self, fake_git_repo, canned_artifact):
        row = self._run(fake_git_repo, canned_artifact, acceptance_exit=1)
        assert row.success is False
        assert row.acceptance_tests_passed is False

    def test_regression_pass_still_success(self, fake_git_repo, canned_artifact):
        row = self._run(
            fake_git_repo, canned_artifact,
            acceptance_exit=0, has_regression_cmd=True, regression_exit=0,
        )
        assert row.success is True
        assert row.regression_tests_passed is True

    def test_regression_fail_means_no_success(self, fake_git_repo, canned_artifact):
        row = self._run(
            fake_git_repo, canned_artifact,
            acceptance_exit=0, has_regression_cmd=True, regression_exit=1,
        )
        assert row.success is False
        assert row.regression_tests_passed is False

    def test_no_regression_cmd_is_null_and_not_failure(self, fake_git_repo, canned_artifact):
        row = self._run(fake_git_repo, canned_artifact, acceptance_exit=0)
        assert row.regression_tests_passed is None
        assert row.success is True  # None regression = not counted as failure

    def test_forbidden_files_changed_means_no_success(self, fake_git_repo, canned_artifact):
        row = self._run(
            fake_git_repo, canned_artifact,
            acceptance_exit=0, corrupt_forbidden=True,
        )
        assert row.forbidden_files_unchanged is False
        assert row.success is False


# ---------------------------------------------------------------------------
# Tests: run_one — sha-mismatch abort: forbidden_files_unchanged is honest
# ---------------------------------------------------------------------------

class TestRunOneShaMismatchForbidden:
    """Fix 2: integrity_failure rows must record forbidden_files_unchanged=True."""

    def test_integrity_failure_sets_forbidden_unchanged_true(
        self, fake_git_repo, canned_task_def
    ):
        """Nothing ran → files are unchanged → forbidden_files_unchanged must be True."""
        platform_repo, _ = fake_git_repo

        bad_artifact = {
            "artifact_id": "art-bad",
            "task_id": "T1",
            "arm": "retrieved",
            "file_sha256": "c" * 64,  # wrong sha
            "sentinel": "mem-bad",
            "memory_markdown": "# MEMORY\nsome content\n",
            "volume_path": "/Volumes/.../MEMORY.md",
        }

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=bad_artifact,
            platform_repo=platform_repo,
            run_agent_fn=lambda *a, **kw: (_ for _ in ()).throw(AssertionError("agent must not run")),
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )

        assert row.failure_reason == "integrity_failure"
        assert row.forbidden_files_unchanged is True, (
            "integrity_failure: agent never ran so forbidden files are unchanged"
        )


# ---------------------------------------------------------------------------
# Tests: run_one — budget breach
# ---------------------------------------------------------------------------

class TestRunOneBudget:

    def test_budget_minutes_sets_failure_reason(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        platform_repo, _ = fake_git_repo

        def stub_timeout(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(stopped_reason="budget_minutes", final_commit=None)

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_timeout,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )

        assert row.failure_reason == "budget_minutes"
        assert row.success is False

    def test_budget_tool_calls_sets_failure_reason(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        platform_repo, _ = fake_git_repo
        # max_tool_calls is 120; agent returns 200 tool calls
        sentinel = canned_artifact["sentinel"]

        def stub_over_budget(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return AgentRunResult(
                trace=[],
                agent_echo=f"ack {sentinel}",
                total_tool_calls=200,  # > 120
                tokens_in=500,
                tokens_out=100,
                elapsed_seconds=5.0,
                final_commit="deadbeef",
                exit_code=0,
                stopped_reason="agent_exit",
                raw_events=[],
            )

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_over_budget,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )

        assert row.failure_reason == "budget_tool_calls"
        assert row.success is False
        assert row.total_tool_calls == 200

    def test_budget_minutes_forbidden_unchanged_honest_no_commits(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """Fix 2: budget_minutes row with no committed changes → forbidden_files_unchanged=True."""
        platform_repo, _ = fake_git_repo

        def stub_timeout(worktree, prompt, model, *, max_minutes, max_tool_calls):
            # Agent timed out without committing anything
            return _make_agent_result(stopped_reason="budget_minutes", final_commit=None)

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_timeout,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )

        assert row.failure_reason == "budget_minutes"
        assert row.forbidden_files_unchanged is True

    def test_budget_minutes_forbidden_unchanged_honest_with_commit(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """Fix 2: budget_minutes row where the agent committed a forbidden file → False."""
        platform_repo, starting_commit = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        def stub_commits_forbidden(worktree, prompt, model, *, max_minutes, max_tool_calls):
            # Commit a change to forbidden.py before "timing out"
            env = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@t.com",
                   "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@t.com"}
            with open(os.path.join(worktree, "forbidden.py"), "w") as f:
                f.write("bad\n")
            subprocess.run(["git", "add", "."], cwd=worktree, env=env, capture_output=True)
            subprocess.run(["git", "commit", "-m", "touch forbidden"],
                           cwd=worktree, env=env, capture_output=True)
            return _make_agent_result(
                sentinel=sentinel, stopped_reason="budget_minutes", final_commit=None
            )

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_commits_forbidden,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )

        assert row.failure_reason == "budget_minutes"
        assert row.forbidden_files_unchanged is False

    def test_budget_minutes_uploads_partial_trace(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """Fix 3: budget_minutes row gets trace_path and final_diff_path from upload helper."""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]
        uploaded = {}

        def stub_upload(local, vol, profile):
            uploaded[vol] = local

        def stub_timeout(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return AgentRunResult(
                trace=[{"type": "read", "target": "a.py"}],
                agent_echo=f"ack {sentinel}",
                total_tool_calls=2,
                tokens_in=10,
                tokens_out=5,
                elapsed_seconds=1800.0,
                final_commit=None,
                exit_code=-1,
                stopped_reason="budget_minutes",
                raw_events=[],
            )

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_timeout,
            write_row_fn=_noop_write,
            upload_fn=stub_upload,
        )

        assert row.failure_reason == "budget_minutes"
        # Both paths should be populated now that we upload from budget-breach paths
        assert row.trace_path is not None
        assert row.final_diff_path is not None
        assert row.trace_path.endswith("transcript.jsonl")
        assert row.final_diff_path.endswith("final.diff")
        # Verify the upload was actually called
        assert any("transcript.jsonl" in k for k in uploaded)
        assert any("final.diff" in k for k in uploaded)

    def test_budget_tool_calls_uploads_partial_trace(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """Fix 3: budget_tool_calls row gets trace_path and final_diff_path."""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]
        uploaded = {}

        def stub_upload(local, vol, profile):
            uploaded[vol] = local

        def stub_over_budget(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return AgentRunResult(
                trace=[{"type": "edit", "target": "src/foo.py"}],
                agent_echo=f"ack {sentinel}",
                total_tool_calls=200,
                tokens_in=500,
                tokens_out=100,
                elapsed_seconds=5.0,
                final_commit="deadbeef",
                exit_code=0,
                stopped_reason="agent_exit",
                raw_events=[],
            )

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_over_budget,
            write_row_fn=_noop_write,
            upload_fn=stub_upload,
        )

        assert row.failure_reason == "budget_tool_calls"
        assert row.trace_path is not None
        assert row.final_diff_path is not None

    def test_upload_failure_prints_warning_with_run_id(
        self, fake_git_repo, canned_artifact, canned_task_def, capsys
    ):
        """Fix 3: upload failure is non-fatal but prints a visible warning with run_id."""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        def exploding_upload(local, vol, profile):
            raise RuntimeError("connection refused")

        def stub_timeout(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(stopped_reason="budget_minutes", sentinel=sentinel)

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_timeout,
            write_row_fn=_noop_write,
            upload_fn=exploding_upload,
        )

        # Upload failure is non-fatal — row still written
        assert row.failure_reason == "budget_minutes"
        # Paths stay None on upload failure
        assert row.trace_path is None
        assert row.final_diff_path is None
        # Warning must be printed with run_id and error text
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert row.run_id in captured.out
        assert "connection refused" in captured.out

    def test_upload_failure_on_success_path_prints_warning(
        self, fake_git_repo, canned_artifact, canned_task_def, capsys
    ):
        """Fix 3: upload failure on the success path also prints a visible warning."""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        def exploding_upload(local, vol, profile):
            raise RuntimeError("disk full")

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(sentinel=sentinel)

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=_noop_write,
            upload_fn=exploding_upload,
        )

        # Run still succeeds
        assert row.success is True
        # Paths stay None
        assert row.trace_path is None
        # Warning printed
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert row.run_id in captured.out


# ---------------------------------------------------------------------------
# Tests: run_one — eval_runs row assembly (all columns)
# ---------------------------------------------------------------------------

class TestRunOneRowAssembly:

    def test_all_not_null_columns_populated(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        trace = [
            {"type": "read", "target": "a.py"},
            {"type": "read", "target": "b.py"},
            {"type": "edit", "target": "src/main.py"},
            {"type": "test", "target": "pytest tests/", "passed": False},
        ]

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return AgentRunResult(
                trace=trace,
                agent_echo=f"ack {sentinel}",
                total_tool_calls=4,
                tokens_in=300,
                tokens_out=80,
                elapsed_seconds=15.0,
                final_commit="cafebabe",
                exit_code=0,
                stopped_reason="agent_exit",
                raw_events=[],
            )

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=2,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )

        # NOT NULL columns must be non-None
        assert row.run_id is not None and len(row.run_id) == 36  # UUID
        assert row.task_id == "T_TEST"
        assert row.arm == "retrieved"
        assert row.repeat_number == 2
        assert row.artifact_id == "art-001"
        assert row.file_sha256 == canned_artifact["file_sha256"]
        assert row.sentinel_expected == sentinel
        assert row.injection_verified is True
        assert row.agent_name == "claude-code"
        assert row.model_name is not None
        assert row.starting_commit is not None
        assert isinstance(row.success, bool)
        assert isinstance(row.acceptance_tests_passed, bool)
        assert isinstance(row.forbidden_files_unchanged, bool)
        assert isinstance(row.started_at, datetime)

        # Populated optional columns
        assert row.elapsed_seconds == 15.0
        assert row.total_tool_calls == 4
        assert row.input_tokens == 300
        assert row.output_tokens == 80
        assert row.final_commit == "cafebabe"

        # Metrics computed from trace
        # 2 reads before first edit to src/main.py
        assert row.exploratory_reads_before_edit == 2
        # 1 failed test cycle
        assert row.failed_test_cycles == 1

        # completed_at must be set on successful path
        assert row.completed_at is not None
        assert isinstance(row.completed_at, datetime)


# ---------------------------------------------------------------------------
# Tests: run_one — T1 oracle overwrite logic
# ---------------------------------------------------------------------------

class TestRunOneT1Oracle:

    def test_oracle_file_written_before_acceptance(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        platform_repo, starting_commit = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        oracle_content = "# oracle test file\ndef test_oracle(): assert True\n"
        oracle_fixture_path = "tests/unit/test_image_selection_auto_k.py"

        oracle_calls = []

        def fake_get_oracle(repo, commit, path):
            oracle_calls.append((repo, commit, path))
            return oracle_content

        # Task with oracle spec
        task_def_with_oracle = {
            **canned_task_def,
            "task_id": "T1",
            "acceptance_oracle": {
                "source_commit": "98b8bd7",
                "fixture_path": oracle_fixture_path,
                "note": "test oracle",
            },
        }

        worktree_state = {}

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            # Check that the oracle file is NOT in the worktree during the run
            oracle_during_run = os.path.join(worktree, oracle_fixture_path)
            worktree_state["oracle_during_run"] = os.path.exists(oracle_during_run)
            worktree_state["worktree"] = worktree
            return _make_agent_result(sentinel=sentinel)

        acceptance_called_at = []

        original_run_cmd = None

        def spy_run_cmd(cmd, cwd, timeout=120):
            # Check if oracle file exists at acceptance time
            if "worktree" in worktree_state:
                oracle_at_acc = os.path.join(
                    worktree_state["worktree"], oracle_fixture_path
                )
                acceptance_called_at.append(os.path.exists(oracle_at_acc))
            # Run the original acceptance command
            import subprocess as sp
            r = sp.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
            return r.returncode == 0, r.stdout + r.stderr

        # Patch _run_cmd in the harness.run module
        with patch("harness.run._run_cmd", side_effect=spy_run_cmd):
            row = run_one(
                task_def=task_def_with_oracle,
                arm="retrieved",
                repeat=1,
                artifact=canned_artifact,
                platform_repo=platform_repo,
                run_agent_fn=stub_agent,
                write_row_fn=_noop_write,
                upload_fn=_noop_upload,
                get_oracle_fn=fake_get_oracle,
            )

        # Oracle function was called with correct args
        assert len(oracle_calls) == 1
        assert oracle_calls[0][1] == "98b8bd7"
        assert oracle_calls[0][2] == oracle_fixture_path

        # Oracle was NOT present during the agent run
        assert worktree_state.get("oracle_during_run") is False

        # Oracle WAS present at acceptance time
        assert len(acceptance_called_at) >= 1
        assert acceptance_called_at[0] is True, "Oracle file must exist at acceptance time"

    def test_oracle_not_triggered_for_non_t1(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        platform_repo, starting_commit = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        oracle_calls = []

        def fake_get_oracle(repo, commit, path):
            oracle_calls.append((commit, path))
            return "oracle content"

        # Use T2 task (no oracle)
        task_def_t2 = {
            **canned_task_def,
            "task_id": "T2",
            # No acceptance_oracle key
        }

        # Need a T2-matching artifact
        t2_artifact = {**canned_artifact, "task_id": "T2"}

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(sentinel=sentinel)

        run_one(
            task_def=task_def_t2,
            arm="retrieved",
            repeat=1,
            artifact=t2_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
            get_oracle_fn=fake_get_oracle,
        )

        assert len(oracle_calls) == 0, "Oracle should not be called for T2"

    def test_oracle_constants_used_as_fallback_defaults(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """Fix 1: T1_ORACLE_COMMIT and T1_ORACLE_FIXTURE_PATH are live fallback defaults.

        When acceptance_oracle omits source_commit / fixture_path, the module
        constants T1_ORACLE_COMMIT and T1_ORACLE_FIXTURE_PATH must be used.
        """
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        oracle_calls = []

        def fake_get_oracle(repo, commit, path):
            oracle_calls.append((commit, path))
            return "# oracle\ndef test_x(): pass\n"

        # acceptance_oracle with NEITHER source_commit nor fixture_path
        task_def_minimal_oracle = {
            **canned_task_def,
            "task_id": "T1",
            "acceptance_oracle": {
                # intentionally empty — should fall back to module constants
                "note": "minimal oracle spec, no source_commit or fixture_path",
            },
        }

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(sentinel=sentinel)

        run_one(
            task_def=task_def_minimal_oracle,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
            get_oracle_fn=fake_get_oracle,
        )

        assert len(oracle_calls) == 1
        commit_used, path_used = oracle_calls[0]
        assert commit_used == T1_ORACLE_COMMIT, (
            f"Expected fallback to T1_ORACLE_COMMIT={T1_ORACLE_COMMIT!r}, "
            f"got {commit_used!r}"
        )
        assert path_used == T1_ORACLE_FIXTURE_PATH, (
            f"Expected fallback to T1_ORACLE_FIXTURE_PATH={T1_ORACLE_FIXTURE_PATH!r}, "
            f"got {path_used!r}"
        )


# ---------------------------------------------------------------------------
# Tests: _check_forbidden_unchanged
# ---------------------------------------------------------------------------

class TestCheckForbiddenUnchanged:

    def test_no_forbidden_files_returns_true(self, fake_git_repo):
        platform_repo, starting_commit = fake_git_repo
        # Create a worktree to test
        wt = os.path.join(os.path.dirname(platform_repo), "wt_forbidden_test")
        try:
            subprocess.run(
                ["git", "worktree", "add", wt, starting_commit],
                cwd=platform_repo, check=True, capture_output=True,
            )
            assert _check_forbidden_unchanged(wt, starting_commit, []) is True
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", wt],
                cwd=platform_repo, capture_output=True,
            )

    def test_no_commits_means_unchanged(self, fake_git_repo):
        platform_repo, starting_commit = fake_git_repo
        wt = os.path.join(os.path.dirname(platform_repo), "wt_forbidden_test2")
        try:
            subprocess.run(
                ["git", "worktree", "add", wt, starting_commit],
                cwd=platform_repo, check=True, capture_output=True,
            )
            assert _check_forbidden_unchanged(wt, starting_commit, ["forbidden.py"]) is True
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", wt],
                cwd=platform_repo, capture_output=True,
            )

    def test_committed_forbidden_file_returns_false(self, fake_git_repo):
        platform_repo, starting_commit = fake_git_repo
        wt = os.path.join(os.path.dirname(platform_repo), "wt_forbidden_test3")
        env = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@t.com",
               "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@t.com"}
        try:
            subprocess.run(
                ["git", "worktree", "add", wt, starting_commit],
                cwd=platform_repo, check=True, capture_output=True,
            )
            # Commit a change to forbidden.py
            (open(os.path.join(wt, "forbidden.py"), "w")).write("bad\n")
            subprocess.run(["git", "add", "."], cwd=wt, env=env, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "touch forbidden"],
                cwd=wt, env=env, capture_output=True,
            )
            assert _check_forbidden_unchanged(wt, starting_commit, ["forbidden.py"]) is False
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", wt],
                cwd=platform_repo, capture_output=True,
            )


# ---------------------------------------------------------------------------
# Tests: Important 1 — acceptance/regression timeout does not crash matrix
# ---------------------------------------------------------------------------

class TestAcceptanceRegressionTimeout:
    """Verify that TimeoutExpired on acceptance/regression is caught and recorded,
    not propagated (Fix: Important 1)."""

    def test_acceptance_timeout_records_failure_reason(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """acceptance_timeout: failure_reason set, row written, no exception raised."""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(sentinel=sentinel)

        def stub_timeout_cmd(cmd, cwd, timeout=120):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        with patch("harness.run._run_cmd", side_effect=stub_timeout_cmd):
            row = run_one(
                task_def=canned_task_def,
                arm="retrieved",
                repeat=1,
                artifact=canned_artifact,
                platform_repo=platform_repo,
                run_agent_fn=stub_agent,
                write_row_fn=_noop_write,
                upload_fn=_noop_upload,
            )

        assert row.failure_reason == "acceptance_timeout"
        assert row.acceptance_tests_passed is False
        assert row.success is False

    def test_acceptance_timeout_does_not_raise(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """TimeoutExpired must be swallowed — run_one must return normally."""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(sentinel=sentinel)

        call_count = [0]

        def stub_timeout_cmd(cmd, cwd, timeout=120):
            call_count[0] += 1
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        # Should not raise anything
        row = None
        with patch("harness.run._run_cmd", side_effect=stub_timeout_cmd):
            row = run_one(
                task_def=canned_task_def,
                arm="retrieved",
                repeat=1,
                artifact=canned_artifact,
                platform_repo=platform_repo,
                run_agent_fn=stub_agent,
                write_row_fn=_noop_write,
                upload_fn=_noop_upload,
            )
        assert row is not None
        assert call_count[0] >= 1  # acceptance command was attempted

    def test_regression_timeout_records_failure_reason(
        self, fake_git_repo, canned_artifact
    ):
        """regression_timeout: failure_reason set when regression cmd times out."""
        platform_repo, starting_commit = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        # acceptance passes, regression times out
        call_count = [0]

        def stub_cmd_with_regression_timeout(cmd, cwd, timeout=120):
            call_count[0] += 1
            if call_count[0] == 1:
                return True, "ok"  # acceptance passes
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)  # regression times out

        task_def_with_regression = {
            "task_id": "T_TEST",
            "goal_prompt": "/goal do something",
            "repository": "~/Projects/platform",
            "starting_commit": starting_commit,
            "acceptance_command": "true",
            "regression_command": "pytest regressions/",
            "required_files": ["src/main.py"],
            "forbidden_files": [],
            "max_minutes": 30,
            "max_tool_calls": 120,
        }

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(sentinel=sentinel)

        with patch("harness.run._run_cmd", side_effect=stub_cmd_with_regression_timeout):
            row = run_one(
                task_def=task_def_with_regression,
                arm="retrieved",
                repeat=1,
                artifact=canned_artifact,
                platform_repo=platform_repo,
                run_agent_fn=stub_agent,
                write_row_fn=_noop_write,
                upload_fn=_noop_upload,
            )

        assert row.failure_reason == "regression_timeout"
        assert row.regression_tests_passed is False
        assert row.success is False

    def test_acceptance_timeout_uses_budget_timeout(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """acceptance timeout is max_minutes*60, not the plumbing 120s default."""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]
        observed_timeouts = []

        def spy_cmd(cmd, cwd, timeout=120):
            observed_timeouts.append(timeout)
            return True, "ok"

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(sentinel=sentinel)

        # canned_task_def has max_minutes=30 → expected timeout = 30*60 = 1800s
        with patch("harness.run._run_cmd", side_effect=spy_cmd):
            run_one(
                task_def=canned_task_def,
                arm="retrieved",
                repeat=1,
                artifact=canned_artifact,
                platform_repo=platform_repo,
                run_agent_fn=stub_agent,
                write_row_fn=_noop_write,
                upload_fn=_noop_upload,
            )

        assert len(observed_timeouts) >= 1
        # Every _run_cmd call in Step 8 must use the budget timeout (1800s for max_minutes=30)
        for t in observed_timeouts:
            assert t == 30 * 60, f"Expected budget timeout 1800s, got {t}s"


# ---------------------------------------------------------------------------
# Tests: Important 2 — run_matrix isolation: one crash must not kill the batch
# ---------------------------------------------------------------------------

class TestRunMatrixIsolation:
    """Verify run_matrix catches exceptions from individual run_one calls and
    continues to the remaining runs (Fix: Important 2)."""

    def test_crash_in_one_run_does_not_abort_others(
        self, fake_git_repo, canned_artifact, tmp_path
    ):
        """When run_one raises for (T1, empty, 1), other cells still run."""
        platform_repo, starting_commit = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        # Build a minimal eval_tasks dir (T_TEST avoids oracle gate)
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        task_def = {
            "task_id": "T_TEST",
            "goal_prompt": "/goal do something",
            "repository": "~/Projects/platform",
            "starting_commit": starting_commit,
            "acceptance_command": f"{sys.executable} -c 'import sys; sys.exit(0)'",
            "regression_command": None,
            "required_files": [],
            "forbidden_files": [],
            "max_minutes": 30,
            "max_tool_calls": 120,
        }
        (tasks_dir / "T_TEST.json").write_text(json.dumps(task_def))

        artifacts = {
            ("T_TEST", "empty"):     {**canned_artifact, "arm": "empty"},
            ("T_TEST", "retrieved"): {**canned_artifact, "arm": "retrieved"},
        }

        # run_matrix iterates arms in order: "empty" first, "retrieved" second.
        # Crash the first call; succeed the second.
        call_count = [0]

        def selective_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated worktree crash on first arm")
            return _make_agent_result(sentinel=sentinel)

        rows = run_matrix(
            task_ids=["T_TEST"],
            arms=["empty", "retrieved"],
            repeats=1,
            eval_tasks_dir=str(tasks_dir),
            platform_repo=platform_repo,
            artifacts_override=artifacts,
            run_agent_fn=selective_agent,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )

        # Only the retrieved arm should have produced a row
        assert len(rows) == 1
        assert rows[0].arm == "retrieved"

    def test_crash_prints_error_line(
        self, fake_git_repo, canned_artifact, tmp_path, capsys
    ):
        """Crash in run_one prints a visible ERROR line with task/arm/repeat."""
        platform_repo, starting_commit = fake_git_repo

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        task_def = {
            "task_id": "T_TEST",
            "goal_prompt": "/goal",
            "repository": "~/Projects/platform",
            "starting_commit": starting_commit,
            "acceptance_command": "true",
            "regression_command": None,
            "required_files": [],
            "forbidden_files": [],
            "max_minutes": 30,
            "max_tool_calls": 120,
        }
        (tasks_dir / "T_TEST.json").write_text(json.dumps(task_def))

        artifacts = {("T_TEST", "empty"): {**canned_artifact, "arm": "empty"}}

        def crashing_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            raise RuntimeError("injected crash")

        run_matrix(
            task_ids=["T_TEST"],
            arms=["empty"],
            repeats=1,
            eval_tasks_dir=str(tasks_dir),
            platform_repo=platform_repo,
            artifacts_override=artifacts,
            run_agent_fn=crashing_agent,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )

        out = capsys.readouterr().out
        assert "ERROR" in out
        assert "T_TEST" in out
        assert "empty" in out


# ---------------------------------------------------------------------------
# Tests: Important 3 — T1 oracle written even when acceptance_oracle key absent
# ---------------------------------------------------------------------------

class TestT1OracleKeyAbsent:
    """Verify the oracle is written for T1 even when 'acceptance_oracle' is not
    present in task_def (Fix: Important 3 — gate is now just `task_id == "T1"`)."""

    def test_oracle_written_when_key_absent(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """T1 without acceptance_oracle key → module constants used, oracle written."""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        oracle_calls = []

        def fake_get_oracle(repo, commit, path):
            oracle_calls.append((commit, path))
            return "# oracle\ndef test_fallback(): pass\n"

        # task_def has NO acceptance_oracle key at all
        task_def_no_oracle_key = {k: v for k, v in canned_task_def.items()
                                   if k != "acceptance_oracle"}
        assert "acceptance_oracle" not in task_def_no_oracle_key
        task_def_no_oracle_key["task_id"] = "T1"

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(sentinel=sentinel)

        run_one(
            task_def=task_def_no_oracle_key,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
            get_oracle_fn=fake_get_oracle,
        )

        # Oracle must have been called with the module constants as defaults
        assert len(oracle_calls) == 1, "Oracle must be written even without acceptance_oracle key"
        commit_used, path_used = oracle_calls[0]
        assert commit_used == T1_ORACLE_COMMIT
        assert path_used == T1_ORACLE_FIXTURE_PATH

    def test_oracle_not_written_during_agent_run(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """Contamination invariant: even without key, oracle is only written AFTER agent."""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        oracle_content = "# oracle after run\n"
        fixture_path = T1_ORACLE_FIXTURE_PATH
        worktree_snapshot = {}

        def fake_get_oracle(repo, commit, path):
            return oracle_content

        task_def_no_key = {k: v for k, v in canned_task_def.items()
                           if k != "acceptance_oracle"}
        task_def_no_key["task_id"] = "T1"

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            # Record whether oracle file exists DURING agent run
            oracle_path = os.path.join(worktree, fixture_path)
            worktree_snapshot["oracle_during_run"] = os.path.exists(oracle_path)
            return _make_agent_result(sentinel=sentinel)

        run_one(
            task_def=task_def_no_key,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
            get_oracle_fn=fake_get_oracle,
        )

        assert worktree_snapshot["oracle_during_run"] is False, (
            "Oracle must not be present during agent run (contamination invariant)"
        )


# ---------------------------------------------------------------------------
# Tests: Fix 1 — mkdir before cp in upload_to_volume
# ---------------------------------------------------------------------------

class TestUploadToVolumeMkdir:
    """Verify upload_to_volume issues mkdir before cp (Fix 1)."""

    def test_mkdir_called_before_cp(self):
        """mkdir must be issued for the parent dir before cp is called.

        CLI command structure: ["databricks", "fs", <subcmd>, <arg>, ...]
        so the subcmd is at index 2.
        """
        from harness.run import upload_to_volume
        import tempfile

        commands_issued = []

        def fake_run(cmd, *, check=False, capture_output=False):
            commands_issued.append(list(cmd))

            class FakeResult:
                returncode = 0
                stdout = b""
                stderr = b""

            return FakeResult()

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            f.write(b"test content\n")
            local = f.name

        try:
            with patch("harness.run.subprocess.run", side_effect=fake_run):
                upload_to_volume(
                    local,
                    "/Volumes/cat/sch/vol/raw/eval_runs/run-abc/transcript.jsonl",
                    "hackathon",
                )
        finally:
            os.unlink(local)

        # subcommand is at index 2: ["databricks", "fs", <subcmd>, ...]
        subcmds = [cmd[2] for cmd in commands_issued if len(cmd) > 2]
        assert "mkdir" in subcmds, f"mkdir must be issued; got subcmds={subcmds}"
        assert "cp" in subcmds, f"cp must be issued; got subcmds={subcmds}"
        # mkdir must come first
        assert subcmds.index("mkdir") < subcmds.index("cp"), (
            "mkdir must precede cp"
        )

    def test_mkdir_targets_parent_dir(self):
        """mkdir target must be the parent directory, not the file path.

        CLI command: ["databricks", "fs", "mkdir", <dir>, "--profile", <profile>]
        so the dir argument is at index 3.
        """
        from harness.run import upload_to_volume
        import tempfile

        mkdir_targets = []

        def fake_run(cmd, *, check=False, capture_output=False):
            if len(cmd) > 2 and cmd[2] == "mkdir":
                mkdir_targets.append(cmd[3])  # the dir argument at index 3

            class FakeResult:
                returncode = 0

            return FakeResult()

        with tempfile.NamedTemporaryFile(suffix=".diff", delete=False) as f:
            f.write(b"diff content\n")
            local = f.name

        try:
            with patch("harness.run.subprocess.run", side_effect=fake_run):
                upload_to_volume(
                    local,
                    "/Volumes/cat/sch/vol/raw/eval_runs/run-xyz/final.diff",
                    "hackathon",
                )
        finally:
            os.unlink(local)

        assert len(mkdir_targets) == 1
        assert mkdir_targets[0] == "dbfs:/Volumes/cat/sch/vol/raw/eval_runs/run-xyz", (
            f"mkdir should target the parent dir, got: {mkdir_targets[0]}"
        )

    def test_upload_run_artifacts_mkdir_called_per_upload(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """_upload_run_artifacts triggers mkdir when the concrete upload_fn calls it."""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        mkdir_calls = []
        cp_calls = []

        def recording_upload(local, vol, profile):
            # Simulate what upload_to_volume does (mkdir then cp) — this is a
            # standalone stub that records calls to verify ordering at the run level.
            import posixpath
            parent = posixpath.dirname(vol)
            mkdir_calls.append(parent)
            cp_calls.append(vol)

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(sentinel=sentinel)

        run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=_noop_write,
            upload_fn=recording_upload,
        )

        # Three files uploaded: transcript.jsonl (raw events), trace.jsonl (reduced), final.diff
        assert len(cp_calls) == 3
        transcript_uploads = [v for v in cp_calls if v.endswith("transcript.jsonl")]
        trace_uploads = [v for v in cp_calls if v.endswith("trace.jsonl")]
        diff_uploads = [v for v in cp_calls if v.endswith("final.diff")]
        assert len(transcript_uploads) == 1
        assert len(trace_uploads) == 1
        assert len(diff_uploads) == 1

        # Parent dir recorded for all three uploads (proves mkdir-before-cp contract
        # is applied at the upload_fn level)
        assert len(mkdir_calls) == 3
        for parent in mkdir_calls:
            # Parent should be the run_id directory, not the file itself
            assert not parent.endswith(".jsonl")
            assert not parent.endswith(".diff")


# ---------------------------------------------------------------------------
# Tests: Fix 2 — SENTINEL_ACK_PREAMBLE wraps goal_prompt uniformly
# ---------------------------------------------------------------------------

class TestSentinelAckPreamble:
    """Verify the effective_prompt passed to run_agent contains the preamble
    + original goal_prompt, and that the sentinel value is NOT embedded
    in the preamble constant (Fix 2)."""

    def test_effective_prompt_contains_preamble_and_goal(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """run_agent receives preamble + original goal_prompt concatenated."""
        from harness.run import SENTINEL_ACK_PREAMBLE
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]
        original_goal = canned_task_def["goal_prompt"]

        prompts_seen = []

        def recording_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            prompts_seen.append(prompt)
            return _make_agent_result(sentinel=sentinel)

        run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=recording_agent,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )

        assert len(prompts_seen) == 1
        effective = prompts_seen[0]

        # Must contain the preamble
        assert SENTINEL_ACK_PREAMBLE in effective, (
            "Preamble must be prepended to the effective_prompt"
        )
        # Must contain the original goal_prompt (unchanged)
        assert original_goal in effective, (
            "Original goal_prompt must appear verbatim in the effective_prompt"
        )
        # Preamble must come BEFORE goal
        assert effective.index(SENTINEL_ACK_PREAMBLE) < effective.index(original_goal), (
            "Preamble must precede goal_prompt"
        )

    def test_preamble_does_not_embed_sentinel_value(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """The SENTINEL_ACK_PREAMBLE constant must not contain the sentinel value.

        The agent must read the sentinel from MEMORY.md, not from the prompt.
        """
        from harness.run import SENTINEL_ACK_PREAMBLE
        sentinel_value = canned_artifact["sentinel"]  # e.g. "mem-T1-retrieved-abc123"

        assert sentinel_value not in SENTINEL_ACK_PREAMBLE, (
            f"Preamble must NOT embed the sentinel value {sentinel_value!r}; "
            "the agent must read it from MEMORY.md"
        )

    def test_preamble_applied_to_all_arms(
        self, fake_git_repo, canned_artifact, canned_task_def, tmp_path
    ):
        """SENTINEL_ACK_PREAMBLE is applied to every arm (empty/retrieved/placebo)."""
        from harness.run import SENTINEL_ACK_PREAMBLE
        platform_repo, starting_commit = fake_git_repo
        sentinel = canned_artifact["sentinel"]
        original_goal = canned_task_def["goal_prompt"]

        # Build a multi-arm artifact map and tasks dir
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        task_def = {**canned_task_def}
        (tasks_dir / f"{task_def['task_id']}.json").write_text(json.dumps(task_def))

        arms_to_test = ["empty", "retrieved", "placebo"]
        artifacts = {
            (task_def["task_id"], arm): {**canned_artifact, "arm": arm}
            for arm in arms_to_test
        }

        prompts_by_arm: dict[str, str] = {}

        def recording_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            # Identify arm from volume path in artifact (via MEMORY.md sentinel — all same here)
            prompts_by_arm[len(prompts_by_arm)] = prompt
            return _make_agent_result(sentinel=sentinel)

        run_matrix(
            task_ids=[task_def["task_id"]],
            arms=arms_to_test,
            repeats=1,
            eval_tasks_dir=str(tasks_dir),
            platform_repo=platform_repo,
            artifacts_override=artifacts,
            run_agent_fn=recording_agent,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )

        assert len(prompts_by_arm) == len(arms_to_test), (
            f"Expected {len(arms_to_test)} agent calls, got {len(prompts_by_arm)}"
        )
        for idx, prompt in prompts_by_arm.items():
            assert SENTINEL_ACK_PREAMBLE in prompt, (
                f"Arm call #{idx}: preamble missing from effective_prompt"
            )
            assert original_goal in prompt, (
                f"Arm call #{idx}: original goal missing from effective_prompt"
            )

    def test_preamble_constant_instructs_read_from_file(self):
        """Preamble must instruct agent to READ from MEMORY.md, not from the prompt."""
        from harness.run import SENTINEL_ACK_PREAMBLE
        # Must reference MEMORY.md by name
        assert "MEMORY.md" in SENTINEL_ACK_PREAMBLE, (
            "Preamble must tell the agent to read from MEMORY.md"
        )
        # Must not contain any sentinel-like pattern (mem- prefix)
        import re
        assert not re.search(r'\bmem-[a-z0-9]', SENTINEL_ACK_PREAMBLE), (
            "Preamble must not embed a mem-... sentinel value"
        )


# ---------------------------------------------------------------------------
# Tests: Fix A — working-tree diff + forbidden check
# ---------------------------------------------------------------------------

class TestWorkingTreeCapture:
    """Fix A: _capture_working_tree_state and _check_forbidden_unchanged use
    git add -A + --cached to catch uncommitted / untracked changes."""

    def _make_wt(self, platform_repo, starting_commit, suffix):
        wt = os.path.join(os.path.dirname(platform_repo), f"wt_{suffix}")
        subprocess.run(
            ["git", "worktree", "add", wt, starting_commit],
            cwd=platform_repo, check=True, capture_output=True,
        )
        return wt

    def _rm_wt(self, platform_repo, wt):
        subprocess.run(
            ["git", "worktree", "remove", "--force", wt],
            cwd=platform_repo, capture_output=True,
        )

    def test_new_untracked_file_appears_in_diff(self, fake_git_repo):
        """Agent creates a NEW untracked file in an allowed dir → captured diff is
        non-empty and the file name appears in the changed set. (Fix A, case 1)"""
        platform_repo, starting_commit = fake_git_repo
        wt = self._make_wt(platform_repo, starting_commit, "untracked_new")
        try:
            # Create a new file in an allowed directory WITHOUT committing
            new_file_rel = "src/new_module.py"
            new_file_abs = os.path.join(wt, new_file_rel)
            os.makedirs(os.path.dirname(new_file_abs), exist_ok=True)
            with open(new_file_abs, "w") as f:
                f.write("# new module created by agent\n")

            diff_text, changed_set = _capture_working_tree_state(wt, starting_commit)

            assert diff_text.strip() != "", (
                "Diff must be non-empty when the agent created a new untracked file"
            )
            assert any("new_module.py" in p for p in changed_set), (
                f"new_module.py must appear in changed_set; got {changed_set}"
            )
        finally:
            self._rm_wt(platform_repo, wt)

    def test_uncommitted_forbidden_file_returns_false(self, fake_git_repo):
        """Agent modifies a forbidden path WITHOUT committing → forbidden_files_unchanged
        is False. This is the core bug — the old committed-only check missed this. (Fix A, case 2)"""
        platform_repo, starting_commit = fake_git_repo
        wt = self._make_wt(platform_repo, starting_commit, "uncommitted_forbidden")
        try:
            # Create forbidden.py in the working tree WITHOUT staging or committing
            forbidden_abs = os.path.join(wt, "forbidden.py")
            with open(forbidden_abs, "w") as f:
                f.write("# forbidden but not committed\n")

            # The old git diff --name-only starting_commit..HEAD would return empty
            # (nothing committed) and incorrectly report forbidden_files_unchanged=True.
            # The new working-tree approach must catch this.
            result = _check_forbidden_unchanged(wt, starting_commit, ["forbidden.py"])
            assert result is False, (
                "Uncommitted modification to a forbidden file must be detected "
                "(this was the core bug: the old committed-only check missed it)"
            )
        finally:
            self._rm_wt(platform_repo, wt)

    def test_only_allowed_paths_changed_passes_forbidden_check(self, fake_git_repo):
        """Only allowed paths changed (no forbidden files touched) → True. (Fix A, case 3)"""
        platform_repo, starting_commit = fake_git_repo
        wt = self._make_wt(platform_repo, starting_commit, "allowed_only")
        try:
            # Create allowed files, leave forbidden.py untouched
            allowed_abs = os.path.join(wt, "src", "allowed.py")
            os.makedirs(os.path.dirname(allowed_abs), exist_ok=True)
            with open(allowed_abs, "w") as f:
                f.write("# allowed file\n")

            result = _check_forbidden_unchanged(wt, starting_commit, ["forbidden.py"])
            assert result is True, (
                "Only allowed paths changed; forbidden check must return True"
            )
        finally:
            self._rm_wt(platform_repo, wt)

    def test_oracle_file_not_in_captured_diff(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """File written AFTER the capture point (the oracle) does NOT appear in the
        captured diff — proves ordering: capture-before-oracle. (Fix A, case 4)"""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        # Use a T1 task so the oracle write is triggered
        task_def_t1 = {**canned_task_def, "task_id": "T1"}

        uploads: dict[str, str] = {}

        def recording_upload(local, vol, profile):
            with open(local) as f:
                uploads[vol] = f.read()

        def fake_oracle(repo, commit, path):
            return "# oracle injected after agent\ndef test_oracle(): pass\n"

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(sentinel=sentinel)

        run_one(
            task_def=task_def_t1,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=_noop_write,
            upload_fn=recording_upload,
            get_oracle_fn=fake_oracle,
        )

        diff_key = next((k for k in uploads if k.endswith("final.diff")), None)
        assert diff_key is not None, "final.diff must have been uploaded"
        diff_text = uploads[diff_key]

        # The oracle fixture path must NOT appear in the agent diff, because the
        # capture happened BEFORE the oracle was written into the worktree.
        assert T1_ORACLE_FIXTURE_PATH not in diff_text, (
            f"Oracle file {T1_ORACLE_FIXTURE_PATH!r} must not appear in the captured "
            f"agent diff — capture must happen before oracle write (ordering bug)"
        )

    def test_run_one_uncommitted_forbidden_caught_in_normal_path(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """run_one: agent creates a forbidden file WITHOUT committing → success=False,
        forbidden_files_unchanged=False. End-to-end proof of Fix A via run_one."""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        def stub_agent_writes_forbidden(worktree, prompt, model, *, max_minutes, max_tool_calls):
            # Write forbidden.py in the working tree but DON'T commit it
            with open(os.path.join(worktree, "forbidden.py"), "w") as f:
                f.write("# forbidden, uncommitted\n")
            return _make_agent_result(sentinel=sentinel)

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent_writes_forbidden,
            write_row_fn=_noop_write,
            upload_fn=_noop_upload,
        )

        assert row.forbidden_files_unchanged is False, (
            "Uncommitted forbidden file must be detected by the working-tree check"
        )
        assert row.success is False


# ---------------------------------------------------------------------------
# Tests: Fix B — raw stream-json uploaded as transcript.jsonl
# ---------------------------------------------------------------------------

class TestRawTranscriptUpload:
    """Fix B: _upload_run_artifacts must upload raw stream-json events as
    transcript.jsonl, not the reduced metric-trace."""

    def test_transcript_contains_raw_event_types(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """transcript.jsonl must contain raw assistant/user/result event dicts,
        NOT reduced trace rows like {"type": "edit"}. (Fix B)"""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        raw_events_stub = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"ack {sentinel}"}],
                },
            },
            {
                "type": "user",
                "message": {"role": "user", "content": []},
            },
            {
                "type": "result",
                "subtype": "success",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        ]

        uploads: dict[str, str] = {}

        def recording_upload(local, vol, profile):
            with open(local) as f:
                uploads[vol] = f.read()

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return AgentRunResult(
                trace=[{"type": "edit", "target": "src/foo.py"}],
                agent_echo=f"ack {sentinel}",
                total_tool_calls=1,
                tokens_in=10,
                tokens_out=5,
                elapsed_seconds=1.0,
                final_commit=None,
                exit_code=0,
                stopped_reason="agent_exit",
                raw_events=raw_events_stub,
            )

        run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=_noop_write,
            upload_fn=recording_upload,
        )

        # transcript.jsonl must have been uploaded
        transcript_key = next(
            (k for k in uploads if k.endswith("transcript.jsonl")), None
        )
        assert transcript_key is not None, "transcript.jsonl must be uploaded"

        # trace.jsonl must also have been uploaded (reduced trace stored separately)
        trace_key = next((k for k in uploads if k.endswith("trace.jsonl")), None)
        assert trace_key is not None, "trace.jsonl must be uploaded"

        # Parse transcript.jsonl lines
        lines = [ln for ln in uploads[transcript_key].splitlines() if ln.strip()]
        assert len(lines) >= 1, "transcript.jsonl must be non-empty"
        events = [json.loads(ln) for ln in lines]

        # Must contain raw stream-json event types
        event_types = {e.get("type") for e in events}
        assert event_types <= {"assistant", "user", "result"}, (
            f"transcript.jsonl must contain raw event types (assistant/user/result); "
            f"got {event_types}"
        )
        # Must NOT contain reduced trace row types
        assert "edit" not in event_types, (
            "transcript.jsonl must not contain reduced trace rows like {type: edit}"
        )
        assert "read" not in event_types
        assert "test" not in event_types

        # trace.jsonl must contain the reduced metric-trace
        trace_lines = [ln for ln in uploads[trace_key].splitlines() if ln.strip()]
        trace_events = [json.loads(ln) for ln in trace_lines]
        trace_types = {e.get("type") for e in trace_events}
        assert "edit" in trace_types, (
            "trace.jsonl must contain reduced trace rows; got no 'edit' row"
        )

    def test_trace_path_points_to_transcript_jsonl(
        self, fake_git_repo, canned_artifact, canned_task_def
    ):
        """EvalRunRow.trace_path must point to transcript.jsonl (raw events). (Fix B)"""
        platform_repo, _ = fake_git_repo
        sentinel = canned_artifact["sentinel"]

        def stub_upload(local, vol, profile):
            pass  # noop but don't raise

        def stub_agent(worktree, prompt, model, *, max_minutes, max_tool_calls):
            return _make_agent_result(sentinel=sentinel, raw_events=[
                {"type": "result", "subtype": "success", "usage": {}}
            ])

        row = run_one(
            task_def=canned_task_def,
            arm="retrieved",
            repeat=1,
            artifact=canned_artifact,
            platform_repo=platform_repo,
            run_agent_fn=stub_agent,
            write_row_fn=_noop_write,
            upload_fn=stub_upload,
        )

        assert row.trace_path is not None
        assert row.trace_path.endswith("transcript.jsonl"), (
            f"trace_path must point to transcript.jsonl (raw events); got {row.trace_path!r}"
        )
