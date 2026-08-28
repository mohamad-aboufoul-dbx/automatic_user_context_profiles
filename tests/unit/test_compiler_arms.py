"""
TDD tests for compiler.arms — Task 4.2.
Write tests FIRST (RED), then implement (GREEN).

Covers:
  - empty  → []
  - static_generic → query-independent, one-per-domain cap, top_k + token_budget caps
  - retrieved → uses goal_embedding as query via select()
  - placebo → only non-repo_domain mems, count equals retrieved count,
               repo_domain never appears, token_budget respected
"""

import pytest
from compiler.arms import pool


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_mem(
    memory_id: str,
    embedding: list,
    confidence: float,
    source_datetime: str,
    token_count: int = 100,
    domain: str = "domain_a",
) -> dict:
    return {
        "memory_id": memory_id,
        "embedding": embedding,
        "confidence": confidence,
        "source_datetime": source_datetime,
        "token_count": token_count,
        "domain": domain,
    }


BASE_CFG = {
    "score_weights": {"semantic": 0.75, "recency": 0.15, "confidence": 0.10},
    "recency_halflife_days": 120,
    "top_k": 10,
    "token_budget": 10_000,
    "tie_breaker": ["confidence_desc", "source_datetime_desc", "memory_id_asc"],
}

NOW = "2026-08-17T00:00:00Z"

REPO_DOMAIN = "repo:platform"


# ---------------------------------------------------------------------------
# arm: empty
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_returns_empty_list_with_mems(self):
        mems = [_make_mem("m1", [1.0, 0.0], 0.9, "2026-08-01T00:00:00Z")]
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        assert pool("empty", mems, task, BASE_CFG, NOW) == []

    def test_returns_empty_list_with_no_mems(self):
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        assert pool("empty", [], task, BASE_CFG, NOW) == []


# ---------------------------------------------------------------------------
# arm: static_generic
# ---------------------------------------------------------------------------

class TestStaticGeneric:

    def test_query_independent(self):
        """Same result regardless of goal_embedding value."""
        mems = [
            _make_mem("m1", [1.0, 0.0], 0.9, "2026-08-01T00:00:00Z", domain="domain_a"),
            _make_mem("m2", [0.0, 1.0], 0.5, "2026-08-01T00:00:00Z", domain="domain_b"),
        ]
        task_x = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        task_y = {"goal_embedding": [0.0, 1.0], "repo_domain": REPO_DOMAIN}
        result_x = pool("static_generic", mems, task_x, BASE_CFG, NOW)
        result_y = pool("static_generic", mems, task_y, BASE_CFG, NOW)
        assert [m["memory_id"] for m in result_x] == [m["memory_id"] for m in result_y]

    def test_ranking_by_confidence_recency_not_semantic(self):
        """static_generic ranks by confidence*recency, not semantic similarity."""
        # high_conf is orthogonal to goal_embedding; low_conf is aligned.
        # With a strong semantic weight in retrieved, low_conf would rank first.
        # static_generic should still prefer high_conf.
        mems = [
            _make_mem("high_conf_orth", [0.0, 1.0], 0.95, "2026-08-15T00:00:00Z", domain="domain_a"),
            _make_mem("low_conf_aligned", [1.0, 0.0], 0.1, "2026-08-15T00:00:00Z", domain="domain_b"),
        ]
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        result = pool("static_generic", mems, task, BASE_CFG, NOW)
        assert result[0]["memory_id"] == "high_conf_orth"

    def test_one_per_domain_cap_keeps_highest_ranked(self):
        """Only the highest-ranked mem per domain is kept."""
        mems = [
            # domain_a: two mems — m1 has higher confidence → m1 wins
            _make_mem("m1", [1.0, 0.0], 0.9, "2026-08-01T00:00:00Z", domain="domain_a"),
            _make_mem("m2", [1.0, 0.0], 0.5, "2026-08-01T00:00:00Z", domain="domain_a"),
            # domain_b: one mem
            _make_mem("m3", [1.0, 0.0], 0.7, "2026-08-01T00:00:00Z", domain="domain_b"),
        ]
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        result = pool("static_generic", mems, task, BASE_CFG, NOW)
        ids = [m["memory_id"] for m in result]
        assert "m2" not in ids   # loser in domain_a
        assert "m1" in ids       # winner in domain_a
        assert "m3" in ids       # only rep of domain_b

    def test_one_per_domain_tie_broken_by_confidence_desc(self):
        """Tie-breaker resolves within-domain competition (confidence_desc)."""
        mems = [
            # same static score (same confidence AND same recency): tie-break by confidence_desc
            # both same date and confidence, so finally memory_id_asc decides
            _make_mem("z_id", [1.0, 0.0], 0.7, "2026-08-01T00:00:00Z", domain="domain_a"),
            _make_mem("a_id", [1.0, 0.0], 0.7, "2026-08-01T00:00:00Z", domain="domain_a"),
        ]
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        result = pool("static_generic", mems, task, BASE_CFG, NOW)
        ids = [m["memory_id"] for m in result]
        assert "a_id" in ids  # wins tie-break (memory_id_asc)
        assert "z_id" not in ids

    def test_top_k_cap(self):
        """top_k cap limits output size."""
        mems = [
            _make_mem(f"m{i}", [1.0, 0.0], float(i) / 10, "2026-08-01T00:00:00Z",
                      domain=f"domain_{i}")
            for i in range(8)
        ]
        cfg = {**BASE_CFG, "top_k": 3}
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        result = pool("static_generic", mems, task, cfg, NOW)
        assert len(result) == 3

    def test_token_budget_cap(self):
        """Stop before adding a record that would exceed token_budget."""
        mems = [
            _make_mem("a", [1.0, 0.0], 0.9, "2026-08-15T00:00:00Z", token_count=300, domain="domain_a"),
            _make_mem("b", [1.0, 0.0], 0.8, "2026-08-14T00:00:00Z", token_count=300, domain="domain_b"),
            _make_mem("c", [1.0, 0.0], 0.7, "2026-08-13T00:00:00Z", token_count=300, domain="domain_c"),
        ]
        cfg = {**BASE_CFG, "token_budget": 700}
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        # a(300) + b(300) = 600 fits; c would make 900 > 700
        result = pool("static_generic", mems, task, cfg, NOW)
        assert [m["memory_id"] for m in result] == ["a", "b"]

    def test_empty_mems_returns_empty(self):
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        assert pool("static_generic", [], task, BASE_CFG, NOW) == []

    def test_deterministic(self):
        """Identical inputs → identical output."""
        mems = [
            _make_mem("m1", [1.0, 0.0], 0.9, "2026-08-01T00:00:00Z", domain="domain_a"),
            _make_mem("m2", [0.0, 1.0], 0.8, "2026-08-01T00:00:00Z", domain="domain_b"),
        ]
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        r1 = pool("static_generic", mems, task, BASE_CFG, NOW)
        r2 = pool("static_generic", mems, task, BASE_CFG, NOW)
        assert [m["memory_id"] for m in r1] == [m["memory_id"] for m in r2]


# ---------------------------------------------------------------------------
# arm: retrieved
# ---------------------------------------------------------------------------

class TestRetrieved:

    def test_uses_goal_embedding_as_query(self):
        """retrieved ranks by semantic similarity to goal_embedding."""
        mems = [
            _make_mem("x_aligned", [1.0, 0.0], 0.5, "2026-08-01T00:00:00Z", domain="domain_a"),
            _make_mem("y_aligned", [0.0, 1.0], 0.5, "2026-08-01T00:00:00Z", domain="domain_b"),
        ]
        # Query aligned with x → x_aligned should rank first
        task_x = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        result_x = pool("retrieved", mems, task_x, BASE_CFG, NOW)
        assert result_x[0]["memory_id"] == "x_aligned"

        # Query aligned with y → y_aligned should rank first
        task_y = {"goal_embedding": [0.0, 1.0], "repo_domain": REPO_DOMAIN}
        result_y = pool("retrieved", mems, task_y, BASE_CFG, NOW)
        assert result_y[0]["memory_id"] == "y_aligned"

    def test_differs_from_static_generic_for_misaligned_embeddings(self):
        """retrieved and static_generic rank differently when semantic != static."""
        # high_conf has high confidence but orthogonal to goal.
        # low_conf is aligned but low confidence.
        mems = [
            _make_mem("high_conf_orth", [0.0, 1.0], 0.95, "2026-08-15T00:00:00Z", domain="domain_a"),
            _make_mem("low_conf_aligned", [1.0, 0.0], 0.1, "2026-08-15T00:00:00Z", domain="domain_b"),
        ]
        # Use a cfg where semantic weight dominates so retrieved prefers aligned mem
        cfg = {
            **BASE_CFG,
            "score_weights": {"semantic": 0.95, "recency": 0.025, "confidence": 0.025},
        }
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}

        static_result = pool("static_generic", mems, task, cfg, NOW)
        retrieved_result = pool("retrieved", mems, task, cfg, NOW)

        # static_generic: confidence*recency wins → high_conf_orth first
        assert static_result[0]["memory_id"] == "high_conf_orth"
        # retrieved: semantic wins → low_conf_aligned first
        assert retrieved_result[0]["memory_id"] == "low_conf_aligned"

    def test_top_k_and_budget_respected(self):
        """retrieved honours top_k and token_budget (via select())."""
        mems = [
            _make_mem(f"m{i}", [1.0, 0.0], float(i) / 10, "2026-08-01T00:00:00Z",
                      domain=f"domain_{i}")
            for i in range(6)
        ]
        cfg = {**BASE_CFG, "top_k": 2}
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        result = pool("retrieved", mems, task, cfg, NOW)
        assert len(result) == 2

    def test_empty_mems_returns_empty(self):
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        assert pool("retrieved", [], task, BASE_CFG, NOW) == []


# ---------------------------------------------------------------------------
# arm: placebo
# ---------------------------------------------------------------------------

class TestPlacebo:

    def test_excludes_repo_domain_mems(self):
        """placebo never includes mems whose domain equals task["repo_domain"]."""
        mems = [
            _make_mem("non_1",   [1.0, 0.0], 0.9, "2026-08-01T00:00:00Z", domain="domain_a"),
            _make_mem("repo_mem",[1.0, 0.0], 0.99,"2026-08-01T00:00:00Z", domain=REPO_DOMAIN),
            _make_mem("non_2",   [1.0, 0.0], 0.8, "2026-08-01T00:00:00Z", domain="domain_b"),
        ]
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        result = pool("placebo", mems, task, BASE_CFG, NOW)
        assert all(m["domain"] != REPO_DOMAIN for m in result)

    def test_repo_domain_never_appears_even_at_highest_confidence(self):
        """repo_domain mem with highest confidence is still excluded."""
        mems = [
            _make_mem("repo_high", [1.0, 0.0], 0.99, "2026-08-17T00:00:00Z", domain=REPO_DOMAIN),
            _make_mem("non_a",     [1.0, 0.0], 0.5,  "2026-08-01T00:00:00Z", domain="domain_a"),
            _make_mem("non_b",     [1.0, 0.0], 0.4,  "2026-08-01T00:00:00Z", domain="domain_b"),
        ]
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        result = pool("placebo", mems, task, BASE_CFG, NOW)
        assert all(m["domain"] != REPO_DOMAIN for m in result)

    def test_count_equals_retrieved_count(self):
        """placebo count equals the number of items retrieved arm selects."""
        mems = [
            _make_mem("non_1", [1.0, 0.0], 0.9, "2026-08-01T00:00:00Z", domain="domain_a"),
            _make_mem("non_2", [1.0, 0.0], 0.8, "2026-08-01T00:00:00Z", domain="domain_b"),
            _make_mem("repo_1",[1.0, 0.0], 0.7, "2026-08-01T00:00:00Z", domain=REPO_DOMAIN),
            _make_mem("non_3", [1.0, 0.0], 0.6, "2026-08-01T00:00:00Z", domain="domain_c"),
        ]
        cfg = {**BASE_CFG, "top_k": 2, "token_budget": 10_000}
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        retrieved_count = len(pool("retrieved", mems, task, cfg, NOW))
        placebo_result = pool("placebo", mems, task, cfg, NOW)
        assert len(placebo_result) == retrieved_count

    def test_count_equals_retrieved_when_retrieved_includes_repo_mem(self):
        """N matches retrieved count exactly even when retrieved selects a repo mem.

        Fixture: 1 repo mem + 3 non-repo mems, top_k=3, large token_budget.
        retrieved picks 3 (repo_1, non_1, non_2); placebo has 3 non-repo mems
        available and must fill all 3 — strict equality, not just <=.
        """
        # retrieved considers ALL mems including repo_domain mems
        mems = [
            _make_mem("repo_1", [1.0, 0.0], 0.95, "2026-08-01T00:00:00Z", domain=REPO_DOMAIN),
            _make_mem("non_1",  [1.0, 0.0], 0.8,  "2026-08-01T00:00:00Z", domain="domain_a"),
            _make_mem("non_2",  [1.0, 0.0], 0.7,  "2026-08-01T00:00:00Z", domain="domain_b"),
            _make_mem("non_3",  [1.0, 0.0], 0.6,  "2026-08-01T00:00:00Z", domain="domain_c"),
        ]
        cfg = {**BASE_CFG, "top_k": 3, "token_budget": 10_000}
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}

        retrieved_count = len(pool("retrieved", mems, task, cfg, NOW))
        placebo_result = pool("placebo", mems, task, cfg, NOW)

        # placebo must not include repo mem
        assert all(m["domain"] != REPO_DOMAIN for m in placebo_result)
        # count must equal retrieved count exactly (3 non-repo mems are available)
        assert len(placebo_result) == retrieved_count

    def test_count_capped_when_non_repo_mems_run_out(self):
        """placebo count equals len(non_repo_mems) when fewer are available than N.

        Fixture: retrieved_count = 3 (top_k=3, 4 total mems including 1 repo).
        Only 2 non-repo mems exist, so placebo is capped at 2, not 3.
        """
        mems = [
            _make_mem("repo_1", [1.0, 0.0], 0.95, "2026-08-01T00:00:00Z", domain=REPO_DOMAIN),
            _make_mem("repo_2", [1.0, 0.0], 0.85, "2026-08-01T00:00:00Z", domain=REPO_DOMAIN),
            _make_mem("non_1",  [1.0, 0.0], 0.7,  "2026-08-01T00:00:00Z", domain="domain_a"),
            _make_mem("non_2",  [1.0, 0.0], 0.6,  "2026-08-01T00:00:00Z", domain="domain_b"),
        ]
        cfg = {**BASE_CFG, "top_k": 3, "token_budget": 10_000}
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}

        retrieved_count = len(pool("retrieved", mems, task, cfg, NOW))
        non_repo_mems = [m for m in mems if m["domain"] != REPO_DOMAIN]
        placebo_result = pool("placebo", mems, task, cfg, NOW)

        # retrieved picks 3; only 2 non-repo mems exist → placebo capped at 2
        assert retrieved_count == 3
        assert len(non_repo_mems) == 2
        assert len(placebo_result) == len(non_repo_mems)
        assert all(m["domain"] != REPO_DOMAIN for m in placebo_result)

    def test_respects_token_budget(self):
        """placebo never exceeds token_budget."""
        mems = [
            _make_mem("non_1", [1.0, 0.0], 0.9, "2026-08-15T00:00:00Z", token_count=300, domain="domain_a"),
            _make_mem("non_2", [1.0, 0.0], 0.8, "2026-08-14T00:00:00Z", token_count=300, domain="domain_b"),
            _make_mem("non_3", [1.0, 0.0], 0.7, "2026-08-13T00:00:00Z", token_count=300, domain="domain_c"),
            _make_mem("repo_1",[1.0, 0.0], 0.6, "2026-08-01T00:00:00Z", token_count=100, domain=REPO_DOMAIN),
        ]
        cfg = {**BASE_CFG, "top_k": 5, "token_budget": 700}
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        result = pool("placebo", mems, task, cfg, NOW)
        total_tokens = sum(m["token_count"] for m in result)
        assert total_tokens <= 700
        assert all(m["domain"] != REPO_DOMAIN for m in result)

    def test_query_independent_ranking(self):
        """placebo uses static confidence*recency ranking, not goal_embedding."""
        mems = [
            # high_conf is orthogonal to goal — static ranking prefers it
            _make_mem("high_conf_orth", [0.0, 1.0], 0.95, "2026-08-15T00:00:00Z", domain="domain_a"),
            _make_mem("low_conf_aligned",[1.0, 0.0], 0.1, "2026-08-15T00:00:00Z", domain="domain_b"),
        ]
        task_x = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        task_y = {"goal_embedding": [0.0, 1.0], "repo_domain": REPO_DOMAIN}

        result_x = pool("placebo", mems, task_x, BASE_CFG, NOW)
        result_y = pool("placebo", mems, task_y, BASE_CFG, NOW)
        # placebo is query-independent: same order for both queries
        assert [m["memory_id"] for m in result_x] == [m["memory_id"] for m in result_y]
        # static ranking prefers high_conf_orth
        assert result_x[0]["memory_id"] == "high_conf_orth"

    def test_empty_when_no_non_repo_mems(self):
        """Returns [] when all mems belong to repo_domain."""
        mems = [
            _make_mem("repo_1", [1.0, 0.0], 0.9, "2026-08-01T00:00:00Z", domain=REPO_DOMAIN),
            _make_mem("repo_2", [1.0, 0.0], 0.8, "2026-08-01T00:00:00Z", domain=REPO_DOMAIN),
        ]
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        result = pool("placebo", mems, task, BASE_CFG, NOW)
        assert result == []

    def test_deterministic(self):
        """Identical inputs produce identical output."""
        mems = [
            _make_mem("non_1", [1.0, 0.0], 0.9, "2026-08-01T00:00:00Z", domain="domain_a"),
            _make_mem("non_2", [0.0, 1.0], 0.8, "2026-08-01T00:00:00Z", domain="domain_b"),
            _make_mem("repo_1",[1.0, 0.0], 0.7, "2026-08-01T00:00:00Z", domain=REPO_DOMAIN),
        ]
        task = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}
        r1 = pool("placebo", mems, task, BASE_CFG, NOW)
        r2 = pool("placebo", mems, task, BASE_CFG, NOW)
        assert [m["memory_id"] for m in r1] == [m["memory_id"] for m in r2]


# ---------------------------------------------------------------------------
# cfg["tie_breaker"] invariant — static arms must honour the same override
# ---------------------------------------------------------------------------

class TestStaticArmTieBreaker:
    """Static arms (static_generic, placebo) must honour cfg["tie_breaker"].

    Regression guard for the "same config compiles every arm" invariant:
    a cfg["tie_breaker"] override must affect static_generic/placebo exactly
    as it affects retrieved (via select()).
    """

    # Construct two mems with mathematically identical primary static scores
    # so that only the tie-breaker decides the winner:
    #
    #   old_high_conf: confidence=0.8, age=30 days (= halflife)
    #       → recency_factor = 0.5^(30/30) = 0.5  → static_score = 0.4
    #   new_low_conf:  confidence=0.4, age=0 days
    #       → recency_factor = 0.5^(0/30)  = 1.0  → static_score = 0.4
    #
    # Default tie_breaker ["confidence_desc", ...]: old_high_conf (0.8) wins.
    # Custom  tie_breaker ["source_datetime_desc", ...]: new_low_conf (newer) wins.

    _NOW = "2026-08-17T00:00:00Z"
    # 2026-08-17 minus 30 days = 2026-07-18
    _BASE_CFG_TB = {
        "score_weights": {"semantic": 0.75, "recency": 0.15, "confidence": 0.10},
        "recency_halflife_days": 30,
        "top_k": 10,
        "token_budget": 10_000,
    }
    _MEMS = [
        _make_mem("old_high_conf", [1.0, 0.0], 0.8, "2026-07-18T00:00:00Z", domain="domain_a"),
        _make_mem("new_low_conf",  [1.0, 0.0], 0.4, "2026-08-17T00:00:00Z", domain="domain_b"),
    ]
    _TASK = {"goal_embedding": [1.0, 0.0], "repo_domain": REPO_DOMAIN}

    def test_default_tie_breaker_confidence_desc_wins(self):
        """With default tie_breaker, confidence_desc resolves: old_high_conf first."""
        result = pool("static_generic", self._MEMS, self._TASK, self._BASE_CFG_TB, self._NOW)
        assert result[0]["memory_id"] == "old_high_conf"

    def test_custom_tie_breaker_source_datetime_desc_changes_winner(self):
        """Non-default tie_breaker reorders static_generic outcome.

        Swapping confidence_desc to second position lets source_datetime_desc
        fire first, so the newer mem (new_low_conf) wins despite lower confidence.
        This locks the invariant: arms.py reads cfg["tie_breaker"] via the
        shared tie_break_key() helper rather than hard-coding the default order.
        """
        custom_cfg = {
            **self._BASE_CFG_TB,
            "tie_breaker": ["source_datetime_desc", "confidence_desc", "memory_id_asc"],
        }
        result = pool("static_generic", self._MEMS, self._TASK, custom_cfg, self._NOW)
        assert result[0]["memory_id"] == "new_low_conf"

    def test_placebo_also_honors_custom_tie_breaker(self):
        """placebo (which uses the same _rank_static) also flips under the custom cfg."""
        # Add a repo_domain mem so placebo has something to filter; it will be excluded.
        mems_with_repo = self._MEMS + [
            _make_mem("repo_mem", [1.0, 0.0], 0.99, "2026-08-17T00:00:00Z", domain=REPO_DOMAIN),
        ]
        custom_cfg = {
            **self._BASE_CFG_TB,
            "tie_breaker": ["source_datetime_desc", "confidence_desc", "memory_id_asc"],
        }
        result = pool("placebo", mems_with_repo, self._TASK, custom_cfg, self._NOW)
        assert result[0]["memory_id"] == "new_low_conf"


# ---------------------------------------------------------------------------
# arm: unknown
# ---------------------------------------------------------------------------

class TestUnknownArm:

    def test_raises_value_error(self):
        """Unknown arm name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown arm"):
            pool("bogus", [], {"goal_embedding": [1.0], "repo_domain": "x"}, BASE_CFG, NOW)
