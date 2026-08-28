"""
compiler.score — retrieval scoring and selection for the memory compiler.

Public surface (Task 4.1):
    score(query_vec, mem, cfg, now) -> float
    select(mems, query_vec, cfg, now) -> list[dict]

Zero third-party dependencies; cosine similarity is computed in pure Python.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two equal-length vectors.

    Returns 0.0 if either vector has zero magnitude (avoids ZeroDivisionError).
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must have equal length: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_utc(ts: str) -> datetime:
    """Parse an ISO 8601 string to a UTC-aware datetime.

    Accepts:
        "2026-08-17T00:00:00Z"       — trailing Z
        "2026-08-17T00:00:00+00:00"  — explicit offset
        "2026-08-17T00:00:00"        — naive, assumed UTC
    """
    # Normalise trailing Z to +00:00 so fromisoformat handles it on Python <3.11
    ts_normalised = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    dt = datetime.fromisoformat(ts_normalised)
    if dt.tzinfo is None:
        # Naive datetime — assume UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# Spec-mandated tie-breaker order; used when cfg omits the "tie_breaker" key.
_DEFAULT_TIE_BREAKER: list[str] = [
    "confidence_desc",
    "source_datetime_desc",
    "memory_id_asc",
]


def tie_break_key(primary: float, mem: dict, criteria: list[str]) -> tuple:
    """Return a sort key for a memory record that honours an ordered criteria list.

    The first element is ``-primary`` so that sorting ascending yields descending
    primary score.  Each criterion in *criteria* appends one element:

        "confidence_desc"       → ``-mem["confidence"]``        (higher wins)
        "source_datetime_desc"  → ``-epoch_seconds``            (newer wins)
        "memory_id_asc"         → ``mem["memory_id"]``          (smaller wins)

    Unrecognised criteria strings are silently skipped to allow forward
    compatibility.

    Args:
        primary: The primary score for this record (higher = better).
        mem:     Memory record dict (must contain ``confidence``,
                 ``source_datetime``, and ``memory_id`` keys).
        criteria: Ordered list of criterion strings to use as tie-breakers.

    Returns:
        A tuple suitable for use as the ``key=`` argument to ``sorted`` /
        ``list.sort`` — smaller tuples will be ranked first (i.e. better).
    """
    keys: list = [-primary]
    for criterion in criteria:
        if criterion == "confidence_desc":
            keys.append(-mem["confidence"])
        elif criterion == "source_datetime_desc":
            # Negate epoch seconds so that a larger (newer) timestamp sorts first.
            keys.append(-_parse_utc(mem["source_datetime"]).timestamp())
        elif criterion == "memory_id_asc":
            keys.append(mem["memory_id"])
    return tuple(keys)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(query_vec: list[float], mem: dict, cfg: dict, now: str) -> float:
    """Score a single memory record for inclusion in MEMORY.md.

    Formula (verbatim from spec):
        semantic * cosine(query_vec, mem.embedding)
        + recency * 0.5 ** (age_days / halflife)
        + confidence_weight * mem.confidence

    Args:
        query_vec: Embedding of the retrieval query.
        mem: Memory record with keys:
            - embedding: list[float]
            - confidence: float in [0, 1]
            - source_datetime: ISO 8601 string (Z or +00:00 or naive-UTC)
        cfg: Compiler config with keys:
            - score_weights: {semantic, recency, confidence}  (floats)
            - recency_halflife_days: int
        now: ISO 8601 string for the evaluation timestamp.

    Returns:
        float score (higher = more relevant).
    """
    weights = cfg["score_weights"]
    w_semantic = weights["semantic"]
    w_recency = weights["recency"]
    w_confidence = weights["confidence"]
    halflife = cfg["recency_halflife_days"]

    now_dt = _parse_utc(now)
    src_dt = _parse_utc(mem["source_datetime"])
    age_seconds = (now_dt - src_dt).total_seconds()
    age_days = age_seconds / 86400.0

    semantic_term = w_semantic * _cosine(query_vec, mem["embedding"])
    recency_term = w_recency * (0.5 ** (age_days / halflife))
    confidence_term = w_confidence * mem["confidence"]

    return semantic_term + recency_term + confidence_term


def select(
    mems: list[dict],
    query_vec: list[float],
    cfg: dict,
    now: str,
) -> list[dict]:
    """Rank and select memory records for a query.

    Applies three stopping conditions in order:
        1. top_k cap
        2. token_budget cap (stop before adding a record that would exceed budget)

    Tie-breaker (applied when two records have the same float score):
        cfg["tie_breaker"] is an ordered list of criterion strings:
            "confidence_desc"       — higher confidence wins
            "source_datetime_desc"  — newer source_datetime wins
            "memory_id_asc"         — lexicographically smaller memory_id wins

    Args:
        mems: List of memory records (each must have memory_id, token_count,
              embedding, confidence, source_datetime).
        query_vec: Embedding of the retrieval query.
        cfg: Compiler config (see score() docstring, plus top_k, token_budget,
             tie_breaker).
        now: ISO 8601 evaluation timestamp string.

    Returns:
        Selected memory records in descending score order.
    """
    if not mems:
        return []

    top_k = cfg["top_k"]
    token_budget = cfg["token_budget"]
    tie_breaker: list[str] = cfg.get("tie_breaker", _DEFAULT_TIE_BREAKER)

    # Compute scores once
    scored: list[tuple[float, dict]] = [
        (score(query_vec, m, cfg, now), m) for m in mems
    ]

    scored.sort(key=lambda item: tie_break_key(item[0], item[1], tie_breaker))

    selected: list[dict] = []
    cumulative_tokens = 0

    for _, mem in scored:
        if len(selected) >= top_k:
            break
        tokens = mem.get("token_count", 0)
        if cumulative_tokens + tokens > token_budget:
            # Greedy rank-ordered fill: stop at the first item that would exceed
            # the budget; no backfill of smaller items that might still fit.
            break
        selected.append(mem)
        cumulative_tokens += tokens

    return selected
