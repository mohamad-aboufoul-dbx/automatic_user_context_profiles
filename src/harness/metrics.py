"""Trace metric extraction — Task 5.2.

Pure, deterministic functions.  No I/O, no third-party dependencies.

Trace schema (offline contract):
    trace: list[dict]  — chronologically ordered agent events
    Each event:  {"type": str, "target": str}
    type in {"read", "search", "edit", "test"}
    "test" events additionally carry {"passed": bool}
"""
from __future__ import annotations

__all__ = ["exploratory_reads_before_edit", "failed_test_cycles"]


def exploratory_reads_before_edit(
    trace: list[dict],
    required_files: list[str],
) -> int:
    """Count read/search events that occur before the first edit to a required file.

    "Required files" is the set of files the coding task actually needs the agent
    to change.  Any read or search event before the agent's first edit to one of
    those files is considered *repository-rediscovery cost*.

    If the agent never edits a required file, every read/search event in the
    trace counts (the agent never made it to the target).

    Parameters
    ----------
    trace:
        List of event dicts in chronological order.
    required_files:
        File paths that count as the edit target for this task.

    Returns
    -------
    int
        Number of read/search events before the first required-file edit, or the
        total number of read/search events when no such edit exists.
    """
    required_set = set(required_files)
    count = 0

    for event in trace:
        event_type = event.get("type")

        # Stop as soon as we see an edit to one of the required files.
        if event_type == "edit" and event.get("target") in required_set:
            return count

        if event_type in ("read", "search"):
            count += 1

    # No edit to a required file was found — all read/search events count.
    return count


def failed_test_cycles(trace: list[dict]) -> int:
    """Count test events that did not pass.

    Parameters
    ----------
    trace:
        List of event dicts in chronological order.

    Returns
    -------
    int
        Number of "test" events whose "passed" field is ``False``.
    """
    return sum(
        1
        for event in trace
        if event.get("type") == "test" and event.get("passed") is False
    )
