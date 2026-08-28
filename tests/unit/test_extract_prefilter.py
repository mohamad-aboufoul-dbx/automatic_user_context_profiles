"""
TDD tests for extract.prefilter — Task 2.1.
Write tests FIRST (RED), then implement (GREEN).
"""

import pytest
from extract.prefilter import eligible_sessions, unresolved_prefix_ids


# ---------------------------------------------------------------------------
# Canonical test from the brief
# ---------------------------------------------------------------------------

def test_prefilter_drops_after_cutoff_and_heldout():
    """From task-2.1-brief: keep-1 survives; drop-cutoff and held-out id are dropped."""
    sessions = [
        {"conversation_id": "keep-1", "ended_at": "2026-08-16T23:59:59Z"},
        {"conversation_id": "drop-cutoff", "ended_at": "2026-08-17T00:00:00Z"},
        {"conversation_id": "d8d5b84d-1ddb-4c04-952a-3d071d5223c6", "ended_at": "2026-06-01T00:00:00Z"},
    ]
    cfg = {
        "cutoff_ts": "2026-08-17T00:00:00Z",
        "heldout_conversation_ids": ["d8d5b84d-1ddb-4c04-952a-3d071d5223c6"],
    }
    kept = [s["conversation_id"] for s in eligible_sessions(sessions, cfg)]
    assert kept == ["keep-1"]


# ---------------------------------------------------------------------------
# Boundary: session ending EXACTLY at cutoff is DROPPED (strict <)
# ---------------------------------------------------------------------------

def test_prefilter_boundary_exact_cutoff_is_dropped():
    """A session whose end time == cutoff_ts is dropped (strict < enforced)."""
    sessions = [
        {"conversation_id": "exact-cutoff", "ended_at": "2026-08-17T00:00:00Z"},
        {"conversation_id": "one-second-before", "ended_at": "2026-08-16T23:59:59Z"},
    ]
    cfg = {
        "cutoff_ts": "2026-08-17T00:00:00Z",
        "heldout_conversation_ids": [],
    }
    kept = [s["conversation_id"] for s in eligible_sessions(sessions, cfg)]
    assert kept == ["one-second-before"]
    assert "exact-cutoff" not in kept


# ---------------------------------------------------------------------------
# Cluster-exclusion set (broader heldout_exclusions.json ids)
# ---------------------------------------------------------------------------

def test_prefilter_cluster_exclusion_set():
    """Sessions in the cluster-exclusion set are dropped even if before cutoff and not in heldout ids."""
    sessions = [
        {"conversation_id": "keep-me", "ended_at": "2026-08-01T00:00:00Z"},
        {"conversation_id": "cluster-exclude-1", "ended_at": "2026-08-01T00:00:00Z"},
        {"conversation_id": "cluster-exclude-2", "ended_at": "2026-08-01T00:00:00Z"},
    ]
    cfg = {
        "cutoff_ts": "2026-08-17T00:00:00Z",
        "heldout_conversation_ids": [],
    }
    exclusions = {"cluster-exclude-1", "cluster-exclude-2"}
    kept = [s["conversation_id"] for s in eligible_sessions(sessions, cfg, exclusions)]
    assert kept == ["keep-me"]


# ---------------------------------------------------------------------------
# Alternate end-time field names: event_datetime, source_datetime
# ---------------------------------------------------------------------------

def test_prefilter_event_datetime_field():
    """Supports 'event_datetime' as the end-time key."""
    sessions = [
        {"conversation_id": "ev-keep", "event_datetime": "2026-08-10T00:00:00Z"},
        {"conversation_id": "ev-drop", "event_datetime": "2026-08-17T00:00:00Z"},
    ]
    cfg = {"cutoff_ts": "2026-08-17T00:00:00Z", "heldout_conversation_ids": []}
    kept = [s["conversation_id"] for s in eligible_sessions(sessions, cfg)]
    assert kept == ["ev-keep"]


def test_prefilter_source_datetime_field():
    """Supports 'source_datetime' as the end-time key."""
    sessions = [
        {"conversation_id": "src-keep", "source_datetime": "2026-08-10T00:00:00Z"},
        {"conversation_id": "src-drop", "source_datetime": "2026-08-17T01:00:00Z"},
    ]
    cfg = {"cutoff_ts": "2026-08-17T00:00:00Z", "heldout_conversation_ids": []}
    kept = [s["conversation_id"] for s in eligible_sessions(sessions, cfg)]
    assert kept == ["src-keep"]


# ---------------------------------------------------------------------------
# Combined: before cutoff but in heldout_conversation_ids → still dropped
# ---------------------------------------------------------------------------

def test_prefilter_heldout_id_before_cutoff_is_dropped():
    """A session that is chronologically before the cutoff but in heldout_conversation_ids is still dropped."""
    sessions = [
        {"conversation_id": "heldout-old", "ended_at": "2026-06-01T00:00:00Z"},
        {"conversation_id": "keep-old", "ended_at": "2026-06-01T00:00:00Z"},
    ]
    cfg = {
        "cutoff_ts": "2026-08-17T00:00:00Z",
        "heldout_conversation_ids": ["heldout-old"],
    }
    kept = [s["conversation_id"] for s in eligible_sessions(sessions, cfg)]
    assert kept == ["keep-old"]


# ---------------------------------------------------------------------------
# Empty input → empty output
# ---------------------------------------------------------------------------

def test_prefilter_empty_sessions():
    cfg = {"cutoff_ts": "2026-08-17T00:00:00Z", "heldout_conversation_ids": []}
    assert eligible_sessions([], cfg) == []


# ---------------------------------------------------------------------------
# Regression (PR #7 CRITICAL 1): a PRE-CUTOFF banned id in full form, passed
# via the cluster-exclusion set, must be dropped even though the cutoff keeps
# it.  This is the case the truncated-prefix bug silently let through.
# ---------------------------------------------------------------------------

def test_prefilter_precutoff_banned_full_id_via_exclusions_is_dropped():
    """A 2025 (pre-cutoff) banned conversation, in FULL id form, is dropped by the exclusions set."""
    sessions = [
        {"conversation_id": "11dd3914-74e4-43d2-b7f5-d3ce44e5992c", "ended_at": "2025-12-12T01:12:12Z"},
        {"conversation_id": "02c8e44d-07fc-42a4-8e80-1ec134a9ea3e", "ended_at": "2025-07-07T18:12:32Z"},
        {"conversation_id": "genuinely-eligible", "ended_at": "2025-09-01T00:00:00Z"},
    ]
    cfg = {"cutoff_ts": "2026-08-17T00:00:00Z", "heldout_conversation_ids": []}
    exclusions = {
        "11dd3914-74e4-43d2-b7f5-d3ce44e5992c",
        "02c8e44d-07fc-42a4-8e80-1ec134a9ea3e",
    }
    kept = [s["conversation_id"] for s in eligible_sessions(sessions, cfg, exclusions)]
    assert kept == ["genuinely-eligible"]


# ---------------------------------------------------------------------------
# unresolved_prefix_ids — freeze-time invariant helper
#   Invariant: no real conversation_id may have a banned id as a STRICT prefix
#   unless the banned id equals it.  A banned id that is a strict prefix of a
#   real id (but not equal) is UNRESOLVED and must be flagged.
# ---------------------------------------------------------------------------

def test_unresolved_prefix_ids_flags_strict_prefix():
    """A truncated banned id that strict-prefix-matches a real id is flagged."""
    real = {"02c8e44d-07fc-42a4-8e80-1ec134a9ea3e", "keep-1"}
    banned = {"02c8e44d"}  # truncated prefix of the real UUID
    assert unresolved_prefix_ids(banned, real) == ["02c8e44d"]


def test_unresolved_prefix_ids_exact_match_not_flagged():
    """A banned id that exactly equals a real id is resolved (not flagged)."""
    real = {"02c8e44d-07fc-42a4-8e80-1ec134a9ea3e"}
    banned = {"02c8e44d-07fc-42a4-8e80-1ec134a9ea3e"}
    assert unresolved_prefix_ids(banned, real) == []


def test_unresolved_prefix_ids_no_match_not_flagged():
    """A banned id that matches nothing in the corpus is not a prefix hazard."""
    real = {"02c8e44d-07fc-42a4-8e80-1ec134a9ea3e", "keep-1"}
    banned = {"ffffffff"}  # prefix of nothing present
    assert unresolved_prefix_ids(banned, real) == []


def test_unresolved_prefix_ids_mixed_returns_sorted():
    """Multiple unresolved prefixes are returned deterministically (sorted)."""
    real = {
        "11dd3914-74e4-43d2-b7f5-d3ce44e5992c",
        "02c8e44d-07fc-42a4-8e80-1ec134a9ea3e",
        "resolved-exactly",
    }
    banned = {"11dd3914", "02c8e44d", "resolved-exactly", "nomatch"}
    assert unresolved_prefix_ids(banned, real) == ["02c8e44d", "11dd3914"]
