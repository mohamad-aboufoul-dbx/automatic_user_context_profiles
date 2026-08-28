"""
Contamination pre-filter for the memory extraction step.

Keeps sessions that are safe to extract memories from — i.e. sessions
that end strictly before the temporal cutoff AND are not in the held-out
or cluster-exclusion sets.

No I/O: callers pass all data as arguments.
"""

from __future__ import annotations

from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> datetime:
    """Parse an ISO-8601 timestamp string into a timezone-aware datetime.

    Tolerates a trailing 'Z' (replaces with '+00:00' for fromisoformat).
    If no UTC offset is present after stripping Z, the timestamp is treated
    as UTC.
    """
    s = ts_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _session_end_time(session: dict) -> datetime:
    """Extract the session end time from a session dict.

    Field priority (first present key wins):
      1. ``ended_at``       — preferred; set by the sessionizer as a per-session summary.
      2. ``event_datetime`` — individual row timestamp; used when sessions are
                              represented as the MAX(event_datetime) row.
      3. ``source_datetime``— legacy alias for event_datetime in some pipeline stages.

    Raises:
        KeyError: if none of the recognised keys are present.
    """
    for key in ("ended_at", "event_datetime", "source_datetime"):
        if key in session and session[key] is not None:
            return _parse_ts(session[key])
    raise KeyError(
        f"No recognised end-time key ('ended_at', 'event_datetime', 'source_datetime') "
        f"found in session with keys: {sorted(session.keys())}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def eligible_sessions(
    sessions: list[dict],
    cfg: dict,
    exclusions: set[str] | None = None,
) -> list[dict]:
    """Filter sessions to only those eligible for memory extraction.

    A session is kept IFF **all three** conditions hold:

    1. **Temporal cutoff** — the session end time is strictly LESS THAN
       ``cfg["cutoff_ts"]``.  A session ending at exactly the cutoff is
       DROPPED (strict ``<``).  End time is read from the first present key
       among ``ended_at``, ``event_datetime``, ``source_datetime``.

    2. **Explicit heldout ids** — ``conversation_id`` is NOT in
       ``cfg["heldout_conversation_ids"]``.

    3. **Cluster-exclusion set** — ``conversation_id`` is NOT in the optional
       ``exclusions`` set (the broader cluster ids from
       ``config/heldout_exclusions.json``).

    Args:
        sessions:   List of session dicts.  Each dict must have
                    ``conversation_id`` and at least one of ``ended_at``,
                    ``event_datetime``, or ``source_datetime``.
        cfg:        Config dict — requires ``"cutoff_ts"`` (ISO-8601 string)
                    and ``"heldout_conversation_ids"`` (list of strings).
        exclusions: Optional set of additional conversation_ids to drop
                    (cluster-level exclusions from heldout_exclusions.json).

    Returns:
        Filtered list of session dicts preserving input order.
    """
    cutoff = _parse_ts(cfg["cutoff_ts"])
    heldout: set[str] = set(cfg.get("heldout_conversation_ids", []))
    extra: set[str] = exclusions if exclusions is not None else set()

    result = []
    for session in sessions:
        cid: str = session["conversation_id"]
        if cid in heldout or cid in extra:
            continue
        end = _session_end_time(session)
        if end < cutoff:
            result.append(session)
    return result
