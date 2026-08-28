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


def unresolved_prefix_ids(
    banned_ids: set[str] | list[str],
    real_ids: set[str] | list[str],
) -> list[str]:
    """Return banned ids that are a STRICT prefix of a real conversation_id.

    The contamination guards (``eligible_sessions`` and the notebook JOINs)
    match ``conversation_id`` by **exact equality**.  A banned id that is only
    a *prefix* of a real conversation_id (e.g. the 8-char ``"02c8e44d"`` stored
    for the full ``"02c8e44d-07fc-…"`` UUID) therefore never matches and the
    banned conversation silently leaks through.

    This helper enforces the freeze-time invariant:

        No real conversation_id may have a banned id as a STRICT prefix unless
        the banned id equals it.

    A banned id that equals a real id (resolved) or that matches nothing in the
    corpus (harmless — it can never appear) is NOT flagged.

    Args:
        banned_ids: The banned/excluded conversation ids (heldout ∪ cluster).
        real_ids:   The distinct conversation_ids present in the raw corpus.

    Returns:
        Sorted list of banned ids that strict-prefix-match at least one real id
        without equaling it.  Empty list means every banned id is either an
        exact match or matches nothing — safe for exact-equality guards.
    """
    real_set = set(real_ids)
    flagged: list[str] = []
    for bid in banned_ids:
        if bid in real_set:
            continue  # exact match — resolved
        if any(rid != bid and rid.startswith(bid) for rid in real_set):
            flagged.append(bid)
    return sorted(flagged)
