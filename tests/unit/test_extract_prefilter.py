"""
TDD tests for extract.prefilter — Task 2.1.
Write tests FIRST (RED), then implement (GREEN).
"""

import pytest
from extract.prefilter import eligible_sessions


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
