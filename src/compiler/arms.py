"""
compiler.arms — per-arm candidate pool selection for the memory compiler.

Public surface (Task 4.2):
    pool(arm, mems, task, cfg, now) -> list[dict]

arm ∈ {"empty", "static_generic", "retrieved", "placebo"}

Depends on compiler.score (Task 4.1) — imports select() and the internal
_parse_utc helper to avoid duplicating age/recency logic.
"""

from __future__ import annotations

# _parse_utc is internal to score.py but we own both modules; importing it
# is the explicit requirement ("reuse score.py's recency/age helper").
from compiler.score import select, _parse_utc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _static_score(mem: dict, cfg: dict, now: str) -> float:
    """Query-independent score: confidence * recency_factor.

    recency_factor = 0.5 ** (age_days / recency_halflife_days)

    Reuses _parse_utc from score.py to avoid duplicating age parsing logic.
    """
    halflife = cfg["recency_halflife_days"]
    now_dt = _parse_utc(now)
    src_dt = _parse_utc(mem["source_datetime"])
    age_days = (now_dt - src_dt).total_seconds() / 86400.0
    recency_factor = 0.5 ** (age_days / halflife)
    return mem["confidence"] * recency_factor


def _static_sort_key(mem: dict, cfg: dict, now: str) -> tuple:
    """Sort key for static ranking: descending static score with tie-breakers.

    Tie-breaker order is consistent with score.py's _DEFAULT_TIE_BREAKER:
        1. confidence_desc
        2. source_datetime_desc (newer first)
        3. memory_id_asc
    """
    s = _static_score(mem, cfg, now)
    return (
        -s,
        -mem["confidence"],
        -_parse_utc(mem["source_datetime"]).timestamp(),
        mem["memory_id"],
    )


def _rank_static(mems: list[dict], cfg: dict, now: str) -> list[dict]:
    """Return mems sorted by static score descending (with tie-breakers)."""
    return sorted(mems, key=lambda m: _static_sort_key(m, cfg, now))


def _apply_budget(ranked: list[dict], n: int, token_budget: int) -> list[dict]:
    """Greedy rank-ordered fill: take up to n items without exceeding token_budget.

    Stops at the first item that would push cumulative token_count over budget
    (consistent with select()'s greedy behaviour — no backfill of smaller items).
    """
    selected: list[dict] = []
    cumulative_tokens = 0
    for mem in ranked:
        if len(selected) >= n:
            break
        tokens = mem.get("token_count", 0)
        if cumulative_tokens + tokens > token_budget:
            break
        selected.append(mem)
        cumulative_tokens += tokens
    return selected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pool(
    arm: str,
    mems: list[dict],
    task: dict,
    cfg: dict,
    now: str,
) -> list[dict]:
    """Return the candidate memory pool for the given experimental arm.

    Args:
        arm: One of "empty", "static_generic", "retrieved", "placebo".
        mems: All available memory records. Each record must have:
            memory_id, embedding, confidence, source_datetime, token_count,
            domain (str), and any other fields expected by score.select().
        task: Task dict with:
            - goal_embedding: list[float]  (query vector; used by "retrieved")
            - repo_domain: str             (target repo domain; excluded by "placebo")
        cfg: Compiler config — same shape as score.select() expects:
            score_weights, recency_halflife_days, top_k, token_budget,
            tie_breaker (optional).
        now: ISO 8601 evaluation timestamp.

    Returns:
        List of selected memory records in the order appropriate for the arm.

    Raises:
        ValueError: If arm is not one of the four recognised values.
    """
    if arm == "empty":
        # Frozen empty context — no memories included.
        return []

    elif arm == "static_generic":
        # Query-INDEPENDENT ranking by confidence * recency_factor.
        # Enforces a one-per-domain cap (keep the highest-ranked mem per domain),
        # then applies top_k and token_budget caps.
        ranked = _rank_static(mems, cfg, now)

        # One-per-domain cap: iterate in ranked order; keep the first per domain.
        seen_domains: set[str] = set()
        deduped: list[dict] = []
        for mem in ranked:
            d = mem["domain"]
            if d not in seen_domains:
                seen_domains.add(d)
                deduped.append(mem)

        return _apply_budget(deduped, cfg["top_k"], cfg["token_budget"])

    elif arm == "retrieved":
        # Semantic retrieval: delegate entirely to score.select() using the
        # task's goal_embedding as the query vector.
        return select(mems, task["goal_embedding"], cfg, now)

    elif arm == "placebo":
        # Query-independent (same static ranking as static_generic) but:
        #   1. Candidate pool excludes the target repo_domain.
        #   2. Takes exactly N items, where N = len(retrieved arm's selection).
        #   3. Respects token_budget.
        repo_domain = task["repo_domain"]

        # Step 1: exclude target repo domain.
        candidates = [m for m in mems if m["domain"] != repo_domain]

        # Step 2: rank by static score.
        ranked = _rank_static(candidates, cfg, now)

        # Step 3: N = count from the retrieved arm (computed over ALL mems).
        n = len(pool("retrieved", mems, task, cfg, now))

        # Step 4: greedy fill up to N items within token_budget.
        return _apply_budget(ranked, n, cfg["token_budget"])

    else:
        raise ValueError(
            f"Unknown arm: {arm!r}. Must be one of: empty, static_generic, retrieved, placebo"
        )
