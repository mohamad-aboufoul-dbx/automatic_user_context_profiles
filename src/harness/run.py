"""
Task 5.1: Controlled run driver — one fixed Claude-Code agent per task × arm × repeat.

Local execution only (NOT submitted via databricks jobs submit).
Outputs go to UC: eval_runs row via SQL statement-execution API,
transcript + diff via databricks fs cp to the raw Volume.

                    ┌────────────────────────────────────────────────────────┐
                    │  run_one(task_def, arm, repeat, artifact, model, ...)    │
                    │  1. git worktree add platform@starting_commit → tmpdir   │
                    │  2. stage_memory(md, worktree) → staged_path             │
                    │  3. sha integrity pre-check (abort on mismatch)          │
                    │  4. run_agent(worktree, goal_prompt, model, …) ← SEAM   │
                    │  5. verify_injection + T1 oracle + score                 │
                    │  6. write eval_runs row (UC) + upload trace+diff (Vol)   │
                    └────────────────────────────────────────────────────────┘

EMPIRICAL NOTES (confirmed at Task 5.3):
  - DEFAULT_MODEL: the exact --model value that pins to databricks-claude-sonnet-4-5;
    exposed as a module constant + run parameter, never buried in subprocess call.
  - stream-json field paths: localized to _parse_* helpers and _TOOL_TRACE_TYPE;
    update those after 5.3 without touching orchestration logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from harness.inject import stage_memory, verify_injection
from harness.metrics import exploratory_reads_before_edit, failed_test_cycles

# ---------------------------------------------------------------------------
# Constants (configurable; do not hardcode empirical model/field values)
# ---------------------------------------------------------------------------

CATALOG = "ai_fde_hackathon_catalog"
SCHEMA = "automatic_user_context_profiles"
EVAL_RUNS_TABLE = f"{CATALOG}.{SCHEMA}.eval_runs"
MEMORY_ARTIFACTS_TABLE = f"{CATALOG}.{SCHEMA}.memory_artifacts"

DEFAULT_WAREHOUSE_ID = "41659c95dacd3bf0"
DEFAULT_PROFILE = "hackathon"
DEFAULT_PLATFORM_REPO = os.path.expanduser("~/Projects/platform")
RUNS_VOLUME_BASE = f"/Volumes/{CATALOG}/{SCHEMA}/raw/eval_runs"

AGENT_NAME = "claude-code"

# EMPIRICAL: exact --model alias confirmed at 5.3; kept here as a default
# override via model= parameter in run_one/run_matrix.
DEFAULT_MODEL = "databricks-claude-sonnet-4-5"

# T1 oracle spec (contamination trust anchor: keep oracle OUT of agent context).
# These are the AUTHORITATIVE fallback defaults used when task_def["acceptance_oracle"]
# omits source_commit / fixture_path.  run_one reads oracle_spec.get("source_commit",
# T1_ORACLE_COMMIT) and oracle_spec.get("fixture_path", T1_ORACLE_FIXTURE_PATH) so both
# constants are live — changing them here changes eval behaviour.
T1_ORACLE_COMMIT = "98b8bd7"
T1_ORACLE_FIXTURE_PATH = "tests/unit/test_image_selection_auto_k.py"

# Sentinel-acknowledgment preamble prepended to every goal_prompt at launch time.
#
# Design constraints (all must hold):
#   (a) DOES NOT embed the sentinel value — the agent reads it from MEMORY.md,
#       so echoing it proves ingestion, not just prompt-echoing.
#   (b) Applied IDENTICALLY to ALL arms (empty / static_generic / retrieved /
#       placebo), keeping the comparison fair.
#   (c) Harness-only wrapper — frozen eval/tasks/*.json files are never modified.
#   (d) Works for the empty arm too (its MEMORY.md still carries a sentinel).
SENTINEL_ACK_PREAMBLE = (
    "Before you begin, read the MEMORY.md file in your working directory "
    "and echo its memory-version sentinel (the `mem-...` value in the "
    "MEMORY_SENTINEL comment or 'acknowledge memory version' line) on its "
    "own line in your first response. "
    "Then complete the following task:\n\n"
)


# ---------------------------------------------------------------------------
# stream-json field paths (EMPIRICAL — localized here for easy 5.3 update)
# ---------------------------------------------------------------------------

# Event type values in stream-json output
_STYPE_ASSISTANT = "assistant"
_STYPE_USER = "user"
_STYPE_RESULT = "result"

# Block type values inside message.content
_BTYPE_TOOL_USE = "tool_use"
_BTYPE_TOOL_RESULT = "tool_result"
_BTYPE_TEXT = "text"

# Tool name → trace type (EMPIRICAL: confirm tool names at 5.3)
_TOOL_TRACE_TYPE: dict[str, str] = {
    "Read": "read",
    "Glob": "search",
    "Grep": "search",
    "LS": "search",
    "Find": "search",
    "WebSearch": "search",
    "WebFetch": "search",
    "Edit": "edit",
    "Write": "edit",
    "MultiEdit": "edit",
    "NotebookEdit": "edit",
    "Bash": "bash",  # handled separately
}

# Bash input field for the command string (EMPIRICAL: confirm at 5.3)
_BASH_CMD_FIELD = "command"

# Keywords that identify a bash call as a test run
_TEST_CMD_KEYWORDS = ("pytest", "python -m pytest", "ruff check", "ruff")


# ---------------------------------------------------------------------------
# Data classes (the seam)
# ---------------------------------------------------------------------------

@dataclass
class AgentRunResult:
    """Result returned by run_agent (the agent-runner seam).

    The concrete adapter (``run_agent``) populates this from a real subprocess;
    tests inject a stub that returns canned instances.
    """
    trace: list[dict]
    """Normalized tool-call events: list[{"type","target"}] or {"type","target","passed"}."""

    agent_echo: str
    """Concatenated text output from the agent (all text blocks across assistant turns)."""

    total_tool_calls: int
    """Raw count of tool_use blocks in the stream (all tool names)."""

    tokens_in: int
    tokens_out: int
    elapsed_seconds: float

    final_commit: Optional[str]
    """git rev-parse HEAD in the worktree after the run (may equal starting_commit)."""

    exit_code: int
    stopped_reason: Optional[str]
    """'budget_minutes' | 'budget_tool_calls' | 'agent_exit' | None"""

    raw_events: list[dict] = field(default_factory=list)
    """Raw parsed stream-json events emitted by the agent subprocess.
    Uploaded as transcript.jsonl verbatim; the adapted metric-trace is separate."""


@dataclass
class EvalRunRow:
    """One row of the eval_runs table (SPEC §3.5)."""

    run_id: str
    task_id: str
    arm: str
    repeat_number: int
    artifact_id: str
    file_sha256: str
    sentinel_expected: str
    sentinel_observed: Optional[str]
    injection_verified: bool
    agent_name: str
    model_name: str
    starting_commit: str
    final_commit: Optional[str]
    success: bool
    acceptance_tests_passed: bool
    regression_tests_passed: Optional[bool]
    forbidden_files_unchanged: bool
    elapsed_seconds: Optional[float]
    total_tool_calls: Optional[int]
    exploratory_reads_before_edit: Optional[int]
    failed_test_cycles: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    trace_path: Optional[str]
    final_diff_path: Optional[str]
    failure_reason: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]


# ---------------------------------------------------------------------------
# stream-json → trace adapter (EMPIRICAL field paths, localized above)
# ---------------------------------------------------------------------------

def _get_bash_command(tool_input: dict) -> str:
    """Extract command string from Bash tool input (EMPIRICAL field name)."""
    return str(tool_input.get(_BASH_CMD_FIELD, ""))


def _is_test_command(cmd: str) -> bool:
    """Heuristic: does this bash command run tests/lint?"""
    return any(kw in cmd for kw in _TEST_CMD_KEYWORDS)


def _is_bash_passed(is_error: bool) -> bool:
    """Determine pass/fail for a bash call from its tool_result.

    EMPIRICAL: Claude Code sets is_error=True for non-zero exit codes.
    Confirm against a real transcript at 5.3.
    """
    return not is_error


def adapt_stream_json_to_trace(events: list[dict]) -> list[dict]:
    """Convert a list of parsed stream-json dicts to the harness trace schema.

    Maps tool_use events to ``{"type","target"}`` (and ``{"passed"}`` for tests).
    Bash calls whose command matches ``_TEST_CMD_KEYWORDS`` become test events;
    other Bash calls are skipped (not relevant to efficiency metrics).

    EMPIRICAL: event structure + field paths confirmed at 5.3.  The mapping logic
    is stable; only the field name constants at the top of this module may change.

    Parameters
    ----------
    events:
        Parsed stream-json lines (each parsed with ``json.loads``).

    Returns
    -------
    list[dict]
        Chronologically ordered trace events matching the harness trace schema.
    """
    # Pass 1: collect (tool_use_id, name, input) from assistant-turn tool_use blocks
    tool_uses: list[tuple[str, str, dict]] = []

    # Pass 2: collect tool_result info keyed by tool_use_id
    tool_results: dict[str, tuple[str, bool]] = {}  # id → (content_str, is_error)

    for event in events:
        etype = event.get("type")

        if etype == _STYPE_ASSISTANT:
            msg = event.get("message", {})
            content = msg.get("content", []) if isinstance(msg, dict) else []
            for block in (content if isinstance(content, list) else []):
                if isinstance(block, dict) and block.get("type") == _BTYPE_TOOL_USE:
                    tool_uses.append((
                        block.get("id", ""),
                        block.get("name", ""),
                        block.get("input", {}),
                    ))

        elif etype == _STYPE_USER:
            msg = event.get("message", {})
            content = msg.get("content", []) if isinstance(msg, dict) else []
            for block in (content if isinstance(content, list) else []):
                if isinstance(block, dict) and block.get("type") == _BTYPE_TOOL_RESULT:
                    tid = block.get("tool_use_id", "")
                    raw = block.get("content", "")
                    content_str = raw if isinstance(raw, str) else str(raw)
                    is_error = bool(block.get("is_error", False))
                    tool_results[tid] = (content_str, is_error)

    # Build trace in tool_use order
    trace: list[dict] = []
    for tool_use_id, name, tool_input in tool_uses:
        trace_type = _TOOL_TRACE_TYPE.get(name)
        if trace_type is None:
            continue  # unknown/skipped tool

        if trace_type == "read":
            target = tool_input.get("file_path", tool_input.get("path", name))
            trace.append({"type": "read", "target": str(target)})

        elif trace_type == "search":
            target = tool_input.get("pattern", tool_input.get("query",
                      tool_input.get("path", name)))
            trace.append({"type": "search", "target": str(target)})

        elif trace_type == "edit":
            target = tool_input.get("file_path", tool_input.get("path", ""))
            trace.append({"type": "edit", "target": str(target)})

        elif trace_type == "bash":
            cmd = _get_bash_command(tool_input)
            if _is_test_command(cmd):
                _, is_error = tool_results.get(tool_use_id, ("", False))
                passed = _is_bash_passed(is_error)
                trace.append({"type": "test", "target": cmd, "passed": passed})
            # Non-test bash calls intentionally omitted from trace

    return trace


def _parse_agent_echo(events: list[dict]) -> str:
    """Concatenate all text blocks from assistant messages.

    EMPIRICAL: message.content[].text field path — confirm at 5.3.
    """
    parts: list[str] = []
    for event in events:
        if event.get("type") != _STYPE_ASSISTANT:
            continue
        msg = event.get("message", {})
        content = msg.get("content", []) if isinstance(msg, dict) else []
        for block in (content if isinstance(content, list) else []):
            if isinstance(block, dict) and block.get("type") == _BTYPE_TEXT:
                text = block.get("text", "")
                if text:
                    parts.append(str(text))
    return "\n".join(parts)


def _parse_usage(events: list[dict]) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from stream events.

    Prefers the 'result' event's usage dict; falls back to any event with usage.
    EMPIRICAL: exact key names ('input_tokens', 'output_tokens') — confirm at 5.3.
    """
    # Prefer the top-level result event
    for event in events:
        if event.get("type") == _STYPE_RESULT:
            usage = event.get("usage", {})
            if isinstance(usage, dict) and (
                "input_tokens" in usage or "output_tokens" in usage
            ):
                return (
                    int(usage.get("input_tokens", 0)),
                    int(usage.get("output_tokens", 0)),
                )
    # Fallback: any event with a usage dict
    for event in events:
        usage = event.get("usage", {})
        if isinstance(usage, dict) and (
            "input_tokens" in usage or "output_tokens" in usage
        ):
            return (
                int(usage.get("input_tokens", 0)),
                int(usage.get("output_tokens", 0)),
            )
    return (0, 0)


def _count_tool_uses(events: list[dict]) -> int:
    """Count raw tool_use blocks across all assistant turns."""
    count = 0
    for event in events:
        if event.get("type") == _STYPE_ASSISTANT:
            msg = event.get("message", {})
            content = msg.get("content", []) if isinstance(msg, dict) else []
            for block in (content if isinstance(content, list) else []):
                if isinstance(block, dict) and block.get("type") == _BTYPE_TOOL_USE:
                    count += 1
    return count


# ---------------------------------------------------------------------------
# Git + subprocess helpers (pure wrappers — injectable for tests)
# ---------------------------------------------------------------------------

def _get_git_head(worktree: str) -> Optional[str]:
    """Return rev-parse HEAD in the worktree, or None on failure."""
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree, capture_output=True, text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else None


def _create_worktree(platform_repo: str, worktree_path: str, commit: str) -> None:
    """git worktree add <worktree_path> <commit> in the platform repo."""
    subprocess.run(
        ["git", "worktree", "add", worktree_path, commit],
        cwd=platform_repo, check=True, capture_output=True,
    )


def _remove_worktree(platform_repo: str, worktree_path: str) -> None:
    """git worktree remove --force <worktree_path> (best-effort; errors ignored)."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", worktree_path],
        cwd=platform_repo, capture_output=True,
    )


def _run_cmd(cmd: str, cwd: str, timeout: int = 120) -> tuple[bool, str]:
    """Run a shell command; return (passed, output). Used for acceptance/regression."""
    r = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode == 0, r.stdout + r.stderr


def _capture_working_tree_state(worktree: str, starting_commit: str) -> tuple[str, set[str]]:
    """Stage all working-tree changes and return (diff_text, changed_paths_set).

    Runs ``git add -A`` once so that untracked new files the agent created are
    included, then captures both the full diff and the changed-path set via
    ``--cached`` against starting_commit.

    IMPORTANT: must be called BEFORE any oracle/test-fixture files are written
    into the worktree, otherwise those files appear in the recorded agent diff.
    """
    subprocess.run(["git", "add", "-A"], cwd=worktree, capture_output=True)
    diff_r = subprocess.run(
        ["git", "diff", "--cached", starting_commit],
        cwd=worktree, capture_output=True, text=True,
    )
    names_r = subprocess.run(
        ["git", "diff", "--cached", "--name-only", starting_commit],
        cwd=worktree, capture_output=True, text=True,
    )
    diff_text = diff_r.stdout if diff_r.returncode == 0 else ""
    changed = set(names_r.stdout.strip().splitlines()) if names_r.returncode == 0 else set()
    return diff_text, changed


def _get_final_diff(worktree: str, starting_commit: str) -> str:
    """Return working-tree diff vs starting_commit, including new untracked files.

    Delegates to ``_capture_working_tree_state`` (runs ``git add -A`` then
    ``git diff --cached <starting_commit>``) so untracked files the agent created
    without committing are included.  The old committed-only approach
    (``git diff starting_commit HEAD``) missed those files.
    """
    diff_text, _ = _capture_working_tree_state(worktree, starting_commit)
    return diff_text


def _check_forbidden_unchanged(
    worktree: str,
    starting_commit: str,
    forbidden_files: list[str],
    *,
    _precomputed_changed: Optional[set[str]] = None,
) -> bool:
    """Return True iff none of forbidden_files appear in the working-tree diff.

    Uses ``git add -A`` + ``git diff --cached --name-only`` to catch uncommitted
    changes including new untracked files the agent may have created.  The old
    ``git diff --name-only starting_commit..HEAD`` was unsound: an agent that
    modified a forbidden file without committing would pass the check.

    Pass ``_precomputed_changed`` (the set from a prior ``_capture_working_tree_state``
    call) to reuse an already-staged capture and avoid a redundant ``git add -A``.
    """
    if not forbidden_files:
        return True
    if _precomputed_changed is None:
        _, _precomputed_changed = _capture_working_tree_state(worktree, starting_commit)
    return not any(f in _precomputed_changed for f in forbidden_files)


def get_oracle_content(platform_repo: str, commit: str, file_path: str) -> str:
    """Read file_path from a specific commit in the platform repo (T1 oracle).

    Keeps oracle content out of the agent's context — called only AFTER the run.
    """
    r = subprocess.run(
        ["git", "show", f"{commit}:{file_path}"],
        cwd=platform_repo, capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"oracle: git show {commit}:{file_path} failed in {platform_repo!r}: {r.stderr}"
        )
    return r.stdout


# ---------------------------------------------------------------------------
# SQL builder (pure — fully unit-testable)
# ---------------------------------------------------------------------------

def _build_insert_sql(row: EvalRunRow) -> str:
    """Build a parameterless INSERT SQL for eval_runs (safe: all inputs are internal)."""

    def s(v: Optional[str]) -> str:
        if v is None:
            return "NULL"
        return "'" + str(v).replace("'", "''") + "'"

    def b(v: Optional[bool]) -> str:
        if v is None:
            return "NULL"
        return "TRUE" if v else "FALSE"

    def n(v) -> str:
        if v is None:
            return "NULL"
        return str(v)

    def t(v: Optional[datetime]) -> str:
        if v is None:
            return "NULL"
        iso = v.isoformat() if isinstance(v, datetime) else str(v)
        return f"CAST('{iso}' AS TIMESTAMP)"

    return (
        f"INSERT INTO {EVAL_RUNS_TABLE}\n"
        "(run_id, task_id, arm, repeat_number, artifact_id, file_sha256,\n"
        " sentinel_expected, sentinel_observed, injection_verified,\n"
        " agent_name, model_name, starting_commit, final_commit,\n"
        " success, acceptance_tests_passed, regression_tests_passed, forbidden_files_unchanged,\n"
        " elapsed_seconds, total_tool_calls, exploratory_reads_before_edit, failed_test_cycles,\n"
        " input_tokens, output_tokens, trace_path, final_diff_path,\n"
        " failure_reason, started_at, completed_at)\n"
        "VALUES (\n"
        f"  {s(row.run_id)}, {s(row.task_id)}, {s(row.arm)}, {n(row.repeat_number)},\n"
        f"  {s(row.artifact_id)}, {s(row.file_sha256)},\n"
        f"  {s(row.sentinel_expected)}, {s(row.sentinel_observed)}, {b(row.injection_verified)},\n"
        f"  {s(row.agent_name)}, {s(row.model_name)}, {s(row.starting_commit)}, {s(row.final_commit)},\n"
        f"  {b(row.success)}, {b(row.acceptance_tests_passed)}, {b(row.regression_tests_passed)}, {b(row.forbidden_files_unchanged)},\n"
        f"  {n(row.elapsed_seconds)}, {n(row.total_tool_calls)}, {n(row.exploratory_reads_before_edit)}, {n(row.failed_test_cycles)},\n"
        f"  {n(row.input_tokens)}, {n(row.output_tokens)}, {s(row.trace_path)}, {s(row.final_diff_path)},\n"
        f"  {s(row.failure_reason)}, {t(row.started_at)}, {t(row.completed_at)}\n"
        ")"
    )


# ---------------------------------------------------------------------------
# UC I/O (gated behind functions — tests replace with no-ops)
# ---------------------------------------------------------------------------

def write_eval_runs_row(row: EvalRunRow, warehouse_id: str, profile: str) -> None:
    """Insert one eval_runs row via the SQL statement-execution API.

    Uses the hackathon workspace SDK client (profile param selects ~/.databrickscfg section).
    """
    # Import lazily to avoid hard dep when running unit tests with stub injection
    from databricks.sdk import WorkspaceClient  # type: ignore
    from databricks.sdk.service.sql import StatementState  # type: ignore

    w = WorkspaceClient(profile=profile)
    sql = _build_insert_sql(row)
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="30s",
    )
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(
            f"eval_runs INSERT failed (run_id={row.run_id}): {resp.status.error}"
        )


def upload_to_volume(local_path: str, volume_path: str, profile: str) -> None:
    """Upload a local file to the UC Volume via ``databricks fs cp``.

    volume_path must be an absolute /Volumes/... path.

    ``fs cp`` does NOT auto-create nested Volume directories, so we run
    ``fs mkdir`` on the parent dir first (idempotent; safe to call when
    the directory already exists).
    """
    dbfs_path = (
        f"dbfs:{volume_path}" if not volume_path.startswith("dbfs:") else volume_path
    )
    # Derive the parent directory (e.g. .../eval_runs/{run_id})
    dbfs_parent = dbfs_path.rsplit("/", 1)[0]
    subprocess.run(
        ["databricks", "fs", "mkdir", dbfs_parent, "--profile", profile],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["databricks", "fs", "cp", local_path, dbfs_path, "--profile", profile],
        check=True,
        capture_output=True,
    )


def read_memory_artifacts(
    task_ids: list[str],
    arms: list[str],
    warehouse_id: str,
    profile: str,
) -> dict[tuple[str, str], dict]:
    """Query memory_artifacts for (task_id, arm) pairs and return as a keyed dict.

    Returns:
        {(task_id, arm): {artifact_id, task_id, arm, file_sha256, sentinel,
                          memory_markdown, volume_path}}
    """
    from databricks.sdk import WorkspaceClient  # type: ignore
    from databricks.sdk.service.sql import StatementState  # type: ignore

    w = WorkspaceClient(profile=profile)

    tids_sql = ", ".join(f"'{t}'" for t in task_ids)
    arms_sql = ", ".join(f"'{a}'" for a in arms)
    sql = (
        "SELECT artifact_id, task_id, arm, file_sha256, sentinel,"
        " memory_markdown, volume_path"
        f" FROM {MEMORY_ARTIFACTS_TABLE}"
        f" WHERE task_id IN ({tids_sql}) AND arm IN ({arms_sql})"
    )
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="30s",
    )
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"memory_artifacts query failed: {resp.status.error}")

    col_names = [c.name for c in (resp.manifest.schema.columns or [])]
    result: dict[tuple[str, str], dict] = {}
    for row in (resp.result.data_array or []):
        rd = dict(zip(col_names, row))
        result[(rd["task_id"], rd["arm"])] = rd
    return result


# ---------------------------------------------------------------------------
# Concrete Claude-Code subprocess adapter (EMPIRICAL)
# ---------------------------------------------------------------------------

def run_agent(
    worktree: str,
    goal_prompt: str,
    model: str,
    *,
    max_minutes: int,
    max_tool_calls: int,
    settings_path: Optional[str] = None,
) -> AgentRunResult:
    """Launch Claude Code as a subprocess and collect stream-json output.

    EMPIRICAL NOTES (CONFIRMED at Task 5.3, 2026-08-28):
      - ``model``: passed via --model; ``--model databricks-claude-sonnet-4-5``
        resolves correctly through this env's ai-gateway (verified: result JSON
        modelUsage == 'databricks-claude-sonnet-4-5'). DEFAULT_MODEL is that id.
      - DO NOT pass --bare: it skips the settings/auth config and the subprocess
        fails with "OAuth session expired and could not be refreshed". Inheriting
        the parent env (ANTHROPIC_BASE_URL gateway + auth) is required.
      - stream-json field paths (_parse_* helpers, _TOOL_TRACE_TYPE): verified
        against a real transcript — tool_use in assistant.message.content[],
        tool_result/is_error in user events, usage in the result event.

    Budget enforcement:
      - max_minutes: subprocess timeout (SIGKILL → stopped_reason='budget_minutes').
      - max_tool_calls: post-hoc count (→ stopped_reason='budget_tool_calls').
    """
    cmd = [
        "claude", "-p", goal_prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--model", model,
        "--permission-mode", "bypassPermissions",
        "--add-dir", worktree,
    ]
    if settings_path:
        cmd += ["--settings", settings_path]

    timeout_seconds = max_minutes * 60
    t0 = datetime.now(timezone.utc)

    stopped_reason: Optional[str] = None
    exit_code = -1
    stdout = ""

    try:
        proc = subprocess.run(
            cmd,
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        exit_code = proc.returncode
        stdout = proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        stopped_reason = "budget_minutes"
        exit_code = -1
        raw = exc.stdout or b""
        stdout = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()

    # Parse stream-json lines
    events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    trace = adapt_stream_json_to_trace(events)
    agent_echo = _parse_agent_echo(events)
    tokens_in, tokens_out = _parse_usage(events)
    total_tool_calls = _count_tool_uses(events)

    # Post-hoc budget check
    if stopped_reason is None:
        if total_tool_calls > max_tool_calls:
            stopped_reason = "budget_tool_calls"
        else:
            stopped_reason = "agent_exit"

    final_commit = _get_git_head(worktree)

    return AgentRunResult(
        trace=trace,
        agent_echo=agent_echo,
        total_tool_calls=total_tool_calls,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        elapsed_seconds=elapsed,
        final_commit=final_commit,
        exit_code=exit_code,
        stopped_reason=stopped_reason,
        raw_events=events,
    )


# ---------------------------------------------------------------------------
# Shared upload helper (called from success path AND budget-breach paths)
# ---------------------------------------------------------------------------

def _upload_run_artifacts(
    run_id: str,
    raw_events: list[dict],
    trace: list[dict],
    final_diff: str,
    runs_volume_base: str,
    profile: str,
    upload_fn: Callable,
) -> tuple[Optional[str], Optional[str]]:
    """Upload transcript.jsonl (raw stream-json), trace.jsonl, and final.diff to the UC Volume.

    transcript.jsonl contains the raw agent stream-json events (one JSON object
    per line) — the full assistant/user/result event stream, suitable for
    interpretability and SPEC §8 recorded-trace fallback.

    trace.jsonl contains the reduced metric-trace rows ({"type","target",...}).

    ``final_diff`` must be the pre-captured working-tree diff (from
    ``_capture_working_tree_state``), gathered BEFORE any oracle files are
    written into the worktree.

    Returns ``(trace_path, final_diff_path)`` on success, or ``(None, None)``
    if the upload fails.  Upload failure is NON-FATAL but prints a visible
    warning including the run_id so it is detectable in run logs.

    Called from both the success path and every budget-breach early-return so
    that partial traces from breached runs are preserved for debugging.
    """
    trace_path: Optional[str] = None
    final_diff_path: Optional[str] = None

    # transcript.jsonl: raw stream-json events, one JSON object per line
    transcript_jsonl = "\n".join(json.dumps(e) for e in raw_events) + "\n"
    # trace.jsonl: reduced metric-trace rows
    trace_jsonl = "\n".join(json.dumps(e) for e in trace) + "\n"

    tmp_transcript: Optional[str] = None
    tmp_trace: Optional[str] = None
    tmp_diff: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(transcript_jsonl)
            tmp_transcript = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(trace_jsonl)
            tmp_trace = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            f.write(final_diff)
            tmp_diff = f.name

        vol_transcript = f"{runs_volume_base}/{run_id}/transcript.jsonl"
        vol_trace = f"{runs_volume_base}/{run_id}/trace.jsonl"
        vol_diff = f"{runs_volume_base}/{run_id}/final.diff"
        upload_fn(tmp_transcript, vol_transcript, profile)
        upload_fn(tmp_trace, vol_trace, profile)
        upload_fn(tmp_diff, vol_diff, profile)
        trace_path = vol_transcript
        final_diff_path = vol_diff
    except Exception as exc:
        print(
            f"[run_one] WARNING: artifact upload failed for run_id={run_id}: {exc}",
            flush=True,
        )
        # trace_path / final_diff_path stay None; failure is observable in logs
    finally:
        for p in (tmp_transcript, tmp_trace, tmp_diff):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    return trace_path, final_diff_path


# ---------------------------------------------------------------------------
# Core per-run orchestration
# ---------------------------------------------------------------------------

def _extract_sentinel_observed(agent_echo: str, expected_sentinel: str) -> Optional[str]:
    """Return the expected sentinel if it appears in agent_echo, else None."""
    return expected_sentinel if expected_sentinel in agent_echo else None


def run_one(
    task_def: dict,
    arm: str,
    repeat: int,
    artifact: dict,
    model: str = DEFAULT_MODEL,
    platform_repo: str = DEFAULT_PLATFORM_REPO,
    *,
    run_agent_fn: Optional[Callable] = None,
    write_row_fn: Optional[Callable] = None,
    upload_fn: Optional[Callable] = None,
    get_oracle_fn: Optional[Callable] = None,
    warehouse_id: str = DEFAULT_WAREHOUSE_ID,
    profile: str = DEFAULT_PROFILE,
    runs_volume_base: str = RUNS_VOLUME_BASE,
) -> EvalRunRow:
    """Execute one (task, arm, repeat) run end-to-end.

    Injectables (replace with stubs in tests):
    - run_agent_fn: the agent subprocess (default: run_agent).
    - write_row_fn: UC eval_runs INSERT (default: write_eval_runs_row).
    - upload_fn: Volume upload (default: upload_to_volume).
    - get_oracle_fn: T1 oracle git-show (default: get_oracle_content).

    Steps:
    1. git worktree add at starting_commit (cleaned up in finally).
    2. Materialize memory_markdown → stage_memory → staged_path.
    3. Pre-run sha integrity check; abort on mismatch.
    4. run_agent_fn (the seam).
    5. verify_injection + T1 oracle overwrite + score.
    6. Upload trace+diff; write eval_runs row.
    """
    # Resolve injectables
    _run_agent = run_agent_fn if run_agent_fn is not None else run_agent
    _write_row = write_row_fn if write_row_fn is not None else write_eval_runs_row
    _upload = upload_fn if upload_fn is not None else upload_to_volume
    _get_oracle = get_oracle_fn if get_oracle_fn is not None else get_oracle_content

    task_id: str = task_def["task_id"]
    starting_commit: str = task_def["starting_commit"]
    required_files: list[str] = task_def.get("required_files", []) or []
    forbidden_files: list[str] = task_def.get("forbidden_files", []) or []
    acceptance_command: str = task_def["acceptance_command"]
    regression_command: Optional[str] = task_def.get("regression_command")
    max_minutes: int = task_def["max_minutes"]
    max_tool_calls: int = task_def["max_tool_calls"]
    goal_prompt: str = task_def["goal_prompt"]

    artifact_id: str = artifact["artifact_id"]
    file_sha256: str = artifact["file_sha256"]
    sentinel: str = artifact["sentinel"]
    memory_markdown: str = artifact["memory_markdown"]

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    def _make_failure_row(
        failure_reason: str,
        sentinel_observed: Optional[str] = None,
        injection_verified: bool = False,
        elapsed_seconds: Optional[float] = None,
        total_tool_calls_val: Optional[int] = None,
        input_tokens_val: Optional[int] = None,
        output_tokens_val: Optional[int] = None,
        final_commit: Optional[str] = None,
        # Honest forbidden-files value: True when nothing ran (pre-agent abort);
        # caller passes the real computed value for budget-breach rows where the
        # agent DID run and may have committed changes.
        forbidden_files_unchanged_val: bool = True,
        trace_path: Optional[str] = None,
        final_diff_path: Optional[str] = None,
    ) -> EvalRunRow:
        return EvalRunRow(
            run_id=run_id,
            task_id=task_id,
            arm=arm,
            repeat_number=repeat,
            artifact_id=artifact_id,
            file_sha256=file_sha256,
            sentinel_expected=sentinel,
            sentinel_observed=sentinel_observed,
            injection_verified=injection_verified,
            agent_name=AGENT_NAME,
            model_name=model,
            starting_commit=starting_commit,
            final_commit=final_commit,
            success=False,
            acceptance_tests_passed=False,
            regression_tests_passed=None,
            forbidden_files_unchanged=forbidden_files_unchanged_val,
            elapsed_seconds=elapsed_seconds,
            total_tool_calls=total_tool_calls_val,
            exploratory_reads_before_edit=None,
            failed_test_cycles=None,
            input_tokens=input_tokens_val,
            output_tokens=output_tokens_val,
            trace_path=trace_path,
            final_diff_path=final_diff_path,
            failure_reason=failure_reason,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )

    worktree_path: Optional[str] = None
    worktree_created = False

    try:
        # ── Step 1: create throwaway worktree ──────────────────────────────
        worktree_path = tempfile.mkdtemp(prefix=f"aucp_{task_id}_{arm}_r{repeat}_")
        # git worktree add accepts an existing empty dir
        _create_worktree(platform_repo, worktree_path, starting_commit)
        worktree_created = True

        # ── Step 2: materialize MEMORY.md + stage ──────────────────────────
        tmp_md_fd, tmp_md_path = tempfile.mkstemp(suffix=".md")
        try:
            with os.fdopen(tmp_md_fd, "w", encoding="utf-8") as fh:
                fh.write(memory_markdown)
            staged_path = stage_memory(tmp_md_path, worktree_path)
        finally:
            try:
                os.unlink(tmp_md_path)
            except OSError:
                pass

        # ── Step 3: pre-run sha integrity check ────────────────────────────
        actual_sha = hashlib.sha256(open(staged_path, "rb").read()).hexdigest()
        if actual_sha != file_sha256:
            # Agent never ran — worktree is clean at starting_commit.
            # forbidden_files_unchanged=True is honest: nothing could have changed.
            row = _make_failure_row(
                "integrity_failure",
                forbidden_files_unchanged_val=True,
            )
            _write_row(row, warehouse_id, profile)
            return row

        # ── Step 4: run the agent ───────────────────────────────────────────
        # Wrap goal_prompt with the uniform sentinel-acknowledgment preamble so
        # the agent reads and echoes the sentinel from MEMORY.md (proving ingestion).
        # SENTINEL_ACK_PREAMBLE is applied identically to ALL arms and does NOT
        # embed the sentinel value (that must come from the agent reading the file).
        effective_prompt = SENTINEL_ACK_PREAMBLE + goal_prompt
        agent_result = _run_agent(
            worktree_path,
            effective_prompt,
            model,
            max_minutes=max_minutes,
            max_tool_calls=max_tool_calls,
        )

        # ── Step 5: sentinel echo + injection verified ──────────────────────
        sentinel_observed = _extract_sentinel_observed(agent_result.agent_echo, sentinel)
        injection_verified = verify_injection(
            agent_result.agent_echo, sentinel, staged_path, file_sha256
        )

        # ── Step 6: budget breach → record what we have and return ──────────
        # The agent DID run for both budget-breach kinds, so we COMPUTE the real
        # forbidden-files state from git and UPLOAD whatever partial trace+diff
        # exists — exactly the data you'd need to diagnose a breached run.
        if agent_result.stopped_reason == "budget_minutes":
            # Capture working-tree state (includes uncommitted / untracked files)
            _final_diff, _changed_set = _capture_working_tree_state(
                worktree_path, starting_commit
            )
            _forbidden_ok = _check_forbidden_unchanged(
                worktree_path, starting_commit, forbidden_files,
                _precomputed_changed=_changed_set,
            )
            _trace_path, _diff_path = _upload_run_artifacts(
                run_id, agent_result.raw_events, agent_result.trace, _final_diff,
                runs_volume_base, profile, _upload,
            )
            row = _make_failure_row(
                "budget_minutes",
                sentinel_observed=sentinel_observed,
                injection_verified=injection_verified,
                elapsed_seconds=agent_result.elapsed_seconds,
                total_tool_calls_val=agent_result.total_tool_calls,
                input_tokens_val=agent_result.tokens_in,
                output_tokens_val=agent_result.tokens_out,
                final_commit=agent_result.final_commit,
                forbidden_files_unchanged_val=_forbidden_ok,
                trace_path=_trace_path,
                final_diff_path=_diff_path,
            )
            _write_row(row, warehouse_id, profile)
            return row

        if agent_result.total_tool_calls > max_tool_calls:
            # Capture working-tree state (includes uncommitted / untracked files)
            _final_diff, _changed_set = _capture_working_tree_state(
                worktree_path, starting_commit
            )
            _forbidden_ok = _check_forbidden_unchanged(
                worktree_path, starting_commit, forbidden_files,
                _precomputed_changed=_changed_set,
            )
            _trace_path, _diff_path = _upload_run_artifacts(
                run_id, agent_result.raw_events, agent_result.trace, _final_diff,
                runs_volume_base, profile, _upload,
            )
            row = _make_failure_row(
                "budget_tool_calls",
                sentinel_observed=sentinel_observed,
                injection_verified=injection_verified,
                elapsed_seconds=agent_result.elapsed_seconds,
                total_tool_calls_val=agent_result.total_tool_calls,
                input_tokens_val=agent_result.tokens_in,
                output_tokens_val=agent_result.tokens_out,
                final_commit=agent_result.final_commit,
                forbidden_files_unchanged_val=_forbidden_ok,
                trace_path=_trace_path,
                final_diff_path=_diff_path,
            )
            _write_row(row, warehouse_id, profile)
            return row

        # ── Pre-capture: working-tree state BEFORE oracle write ─────────────
        # Must happen here — after both budget checks (which return early) but
        # BEFORE the T1 oracle file is written into the worktree (Step 7).
        # git add -A stages new/untracked files; --cached diff vs starting_commit
        # captures the full agent diff including files the agent never committed.
        # Reusing the same staged snapshot for both the diff text and the
        # forbidden-files check avoids a redundant git add -A.
        _final_diff, _changed_set = _capture_working_tree_state(
            worktree_path, starting_commit
        )

        # ── Step 7: T1 oracle overwrite (BEFORE acceptance) ─────────────────
        # Contamination invariant: oracle is kept OUT of agent context during run;
        # written only here, after the run completes.
        # Gate on task_id == "T1" ONLY — do NOT require "acceptance_oracle" key.
        # If the key is absent, fall through to the module-level constants so the
        # fallback is real (not bypassed).  T1_ORACLE_COMMIT / T1_ORACLE_FIXTURE_PATH
        # are the live authoritative defaults.
        if task_id == "T1":
            oracle_spec = task_def.get("acceptance_oracle", {})
            _oracle_commit = oracle_spec.get("source_commit", T1_ORACLE_COMMIT)
            _oracle_fixture = oracle_spec.get("fixture_path", T1_ORACLE_FIXTURE_PATH)
            oracle_content = _get_oracle(platform_repo, _oracle_commit, _oracle_fixture)
            oracle_path = os.path.join(worktree_path, _oracle_fixture)
            os.makedirs(os.path.dirname(oracle_path), exist_ok=True)
            with open(oracle_path, "w", encoding="utf-8") as fh:
                fh.write(oracle_content)

        # ── Step 8: run acceptance + regression + forbidden check ────────────
        # Use a budget-derived timeout so T2/T3 live-workspace acceptance suites
        # don't hit the plumbing default.  Catch TimeoutExpired and treat as FAILED
        # rather than crashing the matrix.
        _suite_timeout = max_minutes * 60
        _step8_failure: Optional[str] = None

        try:
            acceptance_tests_passed, _acc_out = _run_cmd(
                acceptance_command, worktree_path, timeout=_suite_timeout
            )
        except subprocess.TimeoutExpired:
            acceptance_tests_passed = False
            _step8_failure = "acceptance_timeout"

        if regression_command:
            try:
                regression_tests_passed_val, _ = _run_cmd(
                    regression_command, worktree_path, timeout=_suite_timeout
                )
                regression_tests_passed: Optional[bool] = regression_tests_passed_val
            except subprocess.TimeoutExpired:
                regression_tests_passed = False
                _step8_failure = _step8_failure or "regression_timeout"
        else:
            regression_tests_passed = None  # null = no regression cmd = not a failure

        # Use the pre-captured _changed_set (gathered before oracle write) so
        # oracle and acceptance-run artifacts don't pollute the forbidden check.
        forbidden_files_unchanged = _check_forbidden_unchanged(
            worktree_path, starting_commit, forbidden_files,
            _precomputed_changed=_changed_set,
        )

        reg_ok = regression_tests_passed is None or regression_tests_passed
        success = acceptance_tests_passed and reg_ok and forbidden_files_unchanged

        # ── Step 9: compute efficiency metrics ──────────────────────────────
        trace = agent_result.trace
        exp_reads = exploratory_reads_before_edit(trace, required_files)
        fail_cycles = failed_test_cycles(trace)

        # ── Step 10: upload trace + diff via shared helper ───────────────────
        # Pass pre-captured _final_diff so oracle/acceptance files (written after
        # the capture point) are not included in the uploaded agent diff.
        trace_path, final_diff_path = _upload_run_artifacts(
            run_id, agent_result.raw_events, trace, _final_diff,
            runs_volume_base, profile, _upload,
        )

        # ── Step 11: assemble and write eval_runs row ────────────────────────
        completed_at = datetime.now(timezone.utc)
        row = EvalRunRow(
            run_id=run_id,
            task_id=task_id,
            arm=arm,
            repeat_number=repeat,
            artifact_id=artifact_id,
            file_sha256=file_sha256,
            sentinel_expected=sentinel,
            sentinel_observed=sentinel_observed,
            injection_verified=injection_verified,
            agent_name=AGENT_NAME,
            model_name=model,
            starting_commit=starting_commit,
            final_commit=agent_result.final_commit,
            success=success,
            acceptance_tests_passed=acceptance_tests_passed,
            regression_tests_passed=regression_tests_passed,
            forbidden_files_unchanged=forbidden_files_unchanged,
            elapsed_seconds=agent_result.elapsed_seconds,
            total_tool_calls=agent_result.total_tool_calls,
            exploratory_reads_before_edit=exp_reads,
            failed_test_cycles=fail_cycles,
            input_tokens=agent_result.tokens_in,
            output_tokens=agent_result.tokens_out,
            trace_path=trace_path,
            final_diff_path=final_diff_path,
            failure_reason=_step8_failure,  # None on clean run; set on suite timeout
            started_at=started_at,
            completed_at=completed_at,
        )
        _write_row(row, warehouse_id, profile)
        return row

    finally:
        # Always clean up the worktree
        if worktree_path:
            if worktree_created:
                _remove_worktree(platform_repo, worktree_path)
            else:
                # worktree add failed; clean up the empty dir mkdtemp left
                shutil.rmtree(worktree_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------

def run_matrix(
    task_ids: list[str],
    arms: list[str],
    repeats: int,
    eval_tasks_dir: str,
    model: str = DEFAULT_MODEL,
    platform_repo: str = DEFAULT_PLATFORM_REPO,
    warehouse_id: str = DEFAULT_WAREHOUSE_ID,
    profile: str = DEFAULT_PROFILE,
    runs_volume_base: str = RUNS_VOLUME_BASE,
    *,
    artifacts_override: Optional[dict[tuple[str, str], dict]] = None,
    run_agent_fn: Optional[Callable] = None,
    write_row_fn: Optional[Callable] = None,
    upload_fn: Optional[Callable] = None,
    get_oracle_fn: Optional[Callable] = None,
) -> list[EvalRunRow]:
    """Run all (task × arm × repeat) combinations sequentially.

    Parameters
    ----------
    task_ids:
        Task IDs to run (e.g. ['T1', 'T2', 'T3']).
    arms:
        Arm names (e.g. ['empty', 'static_generic', 'retrieved', 'placebo']).
    repeats:
        Number of repeats per (task, arm).
    eval_tasks_dir:
        Path to the directory containing T*.json files.
    artifacts_override:
        If provided, use this dict instead of querying memory_artifacts from UC.
        Useful for testing.
    """
    # Load task definitions
    task_defs: dict[str, dict] = {}
    for tid in task_ids:
        path = os.path.join(eval_tasks_dir, f"{tid}.json")
        with open(path, "r", encoding="utf-8") as fh:
            task_defs[tid] = json.load(fh)

    # Load artifacts (from UC or override)
    if artifacts_override is not None:
        artifacts = artifacts_override
    else:
        artifacts = read_memory_artifacts(task_ids, arms, warehouse_id, profile)

    rows: list[EvalRunRow] = []
    for tid in task_ids:
        task_def = task_defs[tid]
        for arm in arms:
            artifact = artifacts.get((tid, arm))
            if artifact is None:
                print(
                    f"[run_matrix] WARNING: no artifact for ({tid}, {arm}) — skipping",
                    flush=True,
                )
                continue
            for rep in range(1, repeats + 1):
                print(
                    f"[run_matrix] starting run: task={tid} arm={arm} repeat={rep}",
                    flush=True,
                )
                try:
                    row = run_one(
                        task_def=task_def,
                        arm=arm,
                        repeat=rep,
                        artifact=artifact,
                        model=model,
                        platform_repo=platform_repo,
                        run_agent_fn=run_agent_fn,
                        write_row_fn=write_row_fn,
                        upload_fn=upload_fn,
                        get_oracle_fn=get_oracle_fn,
                        warehouse_id=warehouse_id,
                        profile=profile,
                        runs_volume_base=runs_volume_base,
                    )
                    rows.append(row)
                    print(
                        f"[run_matrix] done: run_id={row.run_id} success={row.success}"
                        f" failure_reason={row.failure_reason}",
                        flush=True,
                    )
                except Exception as exc:
                    # One flaky run must never abort the remaining matrix.
                    # Prior rows are already written to UC; log and continue.
                    print(
                        f"[run_matrix] ERROR: run crashed and will be skipped "
                        f"task={tid} arm={arm} repeat={rep}: {exc}",
                        flush=True,
                    )
    return rows
