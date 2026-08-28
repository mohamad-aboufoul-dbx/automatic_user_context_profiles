"""
TDD tests for compiler.score — Task 4.1.
Write tests FIRST (RED), then implement (GREEN).
"""

import pytest
from compiler.score import score, select


# ---------------------------------------------------------------------------
# score() — formula correctness
# ---------------------------------------------------------------------------

def test_score_formula_matches_spec():
    """Exact assertion from the task brief."""
    mem = {
        "embedding": [1.0, 0.0],
        "confidence": 0.5,
        "source_datetime": "2026-08-01T00:00:00Z",
    }
    cfg = {
        "score_weights": {"semantic": 0.75, "recency": 0.15, "confidence": 0.10},
        "recency_halflife_days": 120,
    }
    now = "2026-08-17T00:00:00Z"  # 16 days old
    s = score([1.0, 0.0], mem, cfg, now)
    expected = 0.75 * 1.0 + 0.15 * 0.5 ** (16 / 120) + 0.10 * 0.5
    assert abs(s - expected) < 1e-9


def test_score_orthogonal_vectors():
    """Cosine of orthogonal vectors is 0."""
    mem = {
        "embedding": [0.0, 1.0],
        "confidence": 1.0,
        "source_datetime": "2026-08-17T00:00:00Z",  # age = 0
    }
    cfg = {
        "score_weights": {"semantic": 0.75, "recency": 0.15, "confidence": 0.10},
        "recency_halflife_days": 120,
    }
    now = "2026-08-17T00:00:00Z"
    s = score([1.0, 0.0], mem, cfg, now)
    # cosine = 0, age_days = 0 => recency = 0.15 * 1.0, confidence = 0.10 * 1.0
    expected = 0.75 * 0.0 + 0.15 * 1.0 + 0.10 * 1.0
    assert abs(s - expected) < 1e-9


def test_score_zero_age():
    """age_days=0 means recency term = weight * 1.0."""
    mem = {
        "embedding": [1.0, 0.0],
        "confidence": 0.8,
        "source_datetime": "2026-08-17T00:00:00Z",
    }
    cfg = {
        "score_weights": {"semantic": 0.75, "recency": 0.15, "confidence": 0.10},
        "recency_halflife_days": 120,
    }
    now = "2026-08-17T00:00:00Z"
    s = score([1.0, 0.0], mem, cfg, now)
    expected = 0.75 * 1.0 + 0.15 * 1.0 + 0.10 * 0.8
    assert abs(s - expected) < 1e-9


def test_score_naive_datetime_treated_as_utc():
    """ISO string without Z or offset is treated as UTC."""
    mem = {
        "embedding": [1.0, 0.0],
        "confidence": 0.5,
        "source_datetime": "2026-08-01T00:00:00",  # no Z
    }
    cfg = {
        "score_weights": {"semantic": 0.75, "recency": 0.15, "confidence": 0.10},
        "recency_halflife_days": 120,
    }
    now = "2026-08-17T00:00:00Z"
    s = score([1.0, 0.0], mem, cfg, now)
    expected = 0.75 * 1.0 + 0.15 * 0.5 ** (16 / 120) + 0.10 * 0.5
    assert abs(s - expected) < 1e-9


# ---------------------------------------------------------------------------
# select() — ranking, tie-breaker, top_k, token_budget
# ---------------------------------------------------------------------------

def _make_mem(memory_id, embedding, confidence, source_datetime, token_count=100):
    return {
        "memory_id": memory_id,
        "embedding": embedding,
        "confidence": confidence,
        "source_datetime": source_datetime,
        "token_count": token_count,
    }


BASE_CFG = {
    "score_weights": {"semantic": 0.75, "recency": 0.15, "confidence": 0.10},
    "recency_halflife_days": 120,
    "top_k": 10,
    "token_budget": 10000,
    "tie_breaker": ["confidence_desc", "source_datetime_desc", "memory_id_asc"],
}


def test_select_returns_ranked_by_score():
    """Higher-scored mems appear first."""
    mems = [
        _make_mem("a", [0.0, 1.0], 0.5, "2026-08-01T00:00:00Z"),  # low semantic match
        _make_mem("b", [1.0, 0.0], 0.5, "2026-08-01T00:00:00Z"),  # perfect match
    ]
    query = [1.0, 0.0]
    now = "2026-08-17T00:00:00Z"
    result = select(mems, query, BASE_CFG, now)
    assert [m["memory_id"] for m in result] == ["b", "a"]


def test_select_top_k_cap():
    """Only top_k mems are returned."""
    mems = [
        _make_mem(f"m{i}", [1.0, 0.0], float(i) / 10, "2026-08-01T00:00:00Z")
        for i in range(10)
    ]
    cfg = {**BASE_CFG, "top_k": 3}
    result = select(mems, [1.0, 0.0], cfg, "2026-08-17T00:00:00Z")
    assert len(result) == 3


def test_select_token_budget_cap():
    """Stop adding mems once cumulative token_count would exceed budget."""
    mems = [
        _make_mem("a", [1.0, 0.0], 0.9, "2026-08-15T00:00:00Z", token_count=300),
        _make_mem("b", [1.0, 0.0], 0.8, "2026-08-14T00:00:00Z", token_count=300),
        _make_mem("c", [1.0, 0.0], 0.7, "2026-08-13T00:00:00Z", token_count=300),
        _make_mem("d", [1.0, 0.0], 0.6, "2026-08-12T00:00:00Z", token_count=300),
    ]
    cfg = {**BASE_CFG, "top_k": 10, "token_budget": 700}
    # budget 700: a(300) + b(300) = 600 fits; c would make 900 > 700, stop
    result = select(mems, [1.0, 0.0], cfg, "2026-08-17T00:00:00Z")
    assert [m["memory_id"] for m in result] == ["a", "b"]


def test_select_tie_breaker_confidence_desc():
    """Tie-breaker: higher confidence wins when primary scores are truly tied.

    Construction: set confidence weight to 0 so confidence doesn't feed into the
    primary score; use identical embeddings and source_datetime so all other score
    components are equal.  Primary scores are exactly equal → confidence_desc
    tie-breaker is the deciding factor.
    """
    cfg = {
        **BASE_CFG,
        "score_weights": {"semantic": 0.75, "recency": 0.15, "confidence": 0.0},
    }
    mems = [
        _make_mem("low_conf",  [1.0, 0.0], 0.3, "2026-08-01T00:00:00Z"),
        _make_mem("high_conf", [1.0, 0.0], 0.9, "2026-08-01T00:00:00Z"),
    ]
    result = select(mems, [1.0, 0.0], cfg, "2026-08-17T00:00:00Z")
    assert result[0]["memory_id"] == "high_conf"


def test_select_tie_breaker_source_datetime_desc():
    """Tie-breaker: newer source_datetime wins when scores and confidence are tied.

    Construction: set recency weight to 0 so source_datetime doesn't feed into the
    primary score; use identical embeddings and confidence so all other score
    components are equal.  Primary scores are exactly equal, confidence_desc sees
    equal confidence → source_datetime_desc is the deciding factor.
    """
    cfg = {
        **BASE_CFG,
        "score_weights": {"semantic": 0.75, "recency": 0.0, "confidence": 0.10},
    }
    mems = [
        _make_mem("older", [1.0, 0.0], 0.5, "2026-07-01T00:00:00Z"),
        _make_mem("newer", [1.0, 0.0], 0.5, "2026-08-01T00:00:00Z"),
    ]
    result = select(mems, [1.0, 0.0], cfg, "2026-08-17T00:00:00Z")
    assert result[0]["memory_id"] == "newer"


def test_select_tie_breaker_memory_id_asc():
    """When confidence and datetime are tied, lexicographically smaller memory_id wins."""
    mems = [
        _make_mem("z_id", [1.0, 0.0], 0.5, "2026-08-01T00:00:00Z"),
        _make_mem("a_id", [1.0, 0.0], 0.5, "2026-08-01T00:00:00Z"),
    ]
    result = select(mems, [1.0, 0.0], BASE_CFG, "2026-08-17T00:00:00Z")
    assert result[0]["memory_id"] == "a_id"


def test_select_default_tie_breaker_without_cfg_key():
    """select() uses _DEFAULT_TIE_BREAKER when 'tie_breaker' is absent from cfg.

    Verifies the mandated default ["confidence_desc","source_datetime_desc","memory_id_asc"]
    is applied rather than an empty list.  With an empty list, Python's stable sort
    would preserve input order (z_id first) — the assertion would fail, catching
    any regression to cfg.get("tie_breaker", []).
    """
    cfg_no_tb = {k: v for k, v in BASE_CFG.items() if k != "tie_breaker"}
    # All score components equal → primary score tied → default tie-breaker fires
    mems = [
        _make_mem("z_id", [1.0, 0.0], 0.5, "2026-08-01T00:00:00Z"),
        _make_mem("a_id", [1.0, 0.0], 0.5, "2026-08-01T00:00:00Z"),
    ]
    result = select(mems, [1.0, 0.0], cfg_no_tb, "2026-08-17T00:00:00Z")
    # memory_id_asc (last in default) resolves the tie: "a_id" < "z_id"
    assert result[0]["memory_id"] == "a_id"


def test_select_empty_list():
    """select() on empty input returns empty list."""
    result = select([], [1.0, 0.0], BASE_CFG, "2026-08-17T00:00:00Z")
    assert result == []


def test_select_order_preserved_in_output():
    """Selected mems are returned in descending score order."""
    query = [1.0, 0.0]
    mems = [
        _make_mem("low",  [0.0, 1.0], 0.1, "2026-08-01T00:00:00Z"),
        _make_mem("high", [1.0, 0.0], 0.9, "2026-08-01T00:00:00Z"),
        _make_mem("mid",  [0.7, 0.7], 0.5, "2026-08-01T00:00:00Z"),
    ]
    result = select(mems, query, BASE_CFG, "2026-08-17T00:00:00Z")
    scores = [score(query, m, BASE_CFG, "2026-08-17T00:00:00Z") for m in result]
    # Verify each score >= the next
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
