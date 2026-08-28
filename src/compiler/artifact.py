"""
compiler.artifact — helpers for notebook 04 (compile/freeze artifacts).

Public surface (Task 4.4):
    token_count_from_text(text) -> int
    make_artifact_id(task_id, arm, file_sha256) -> str
    spark_row_to_mem(row) -> dict

These are thin, pure-Python helpers kept here so the notebook stays a thin
wiring layer and the logic is unit-testable without Spark.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone


def token_count_from_text(text: str) -> int:
    """Return the whitespace-token count of text.

    Uses str.split() with no arguments — splits on any whitespace and
    strips leading/trailing whitespace — identical to render.py's own
    token estimator so the two stay consistent.

    Args:
        text: Any string (including empty string).

    Returns:
        Number of whitespace-separated tokens (0 for empty/whitespace-only).
    """
    return len(text.split())


def make_artifact_id(task_id: str, arm: str, file_sha256: str) -> str:
    """Deterministic artifact_id: SHA-256(task_id|arm|file_sha256).

    The input is a pipe-delimited string encoded as UTF-8.  Identical
    inputs always produce the same 64-char hex digest, so re-running
    the compiler with the same content yields the same artifact_id —
    enabling the MERGE ON artifact_id idempotency guarantee.

    Args:
        task_id:     Eval task identifier (e.g. "T1").
        arm:         Experimental arm label (e.g. "retrieved").
        file_sha256: SHA-256 hexdigest of the rendered MEMORY.md bytes.

    Returns:
        64-character lowercase hex string.
    """
    raw = f"{task_id}|{arm}|{file_sha256}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def spark_row_to_mem(row: object) -> dict:
    """Convert an atomic_memories Spark Row to a compiler-ready mem dict.

    Bridges the gap between the Spark table schema and the compiler
    interface (pool + render expect a plain Python dict).

    Gap closures applied:
    - Injects ``token_count = token_count_from_text(memory_text)`` because
      the ``atomic_memories`` table has no token_count column.
    - Converts ``embedding`` (Spark ARRAY<FLOAT>) to ``list[float]``.
    - Converts ``source_datetime`` (Spark Timestamp / datetime) to an
      ISO-8601 UTC string ending in "Z" (e.g. "2026-08-10T12:00:00Z").
    - Passes ``confidence`` through as ``float``.
    - Includes ``conversation_id`` for the post-selection contamination
      re-assert (the compiler itself never reads this field).

    Args:
        row: Any dict-subscriptable object with the atomic_memories columns
             memory_id, memory_text, memory_type, domain, embedding,
             confidence, source_datetime, conversation_id.
             In production this is a PySpark Row; in tests a plain dict works.

    Returns:
        Dict with keys: memory_id, memory_text, memory_type, domain,
        embedding (list[float]), confidence (float), source_datetime (str),
        token_count (int), conversation_id (str).
    """
    memory_text: str = row["memory_text"]

    src_dt = row["source_datetime"]
    if isinstance(src_dt, datetime):
        if src_dt.tzinfo is None:
            src_dt = src_dt.replace(tzinfo=timezone.utc)
        source_datetime_str: str = src_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        source_datetime_str = str(src_dt)

    return {
        "memory_id":       row["memory_id"],
        "memory_text":     memory_text,
        "memory_type":     row["memory_type"],
        "domain":          row["domain"],
        "embedding":       [float(x) for x in row["embedding"]],
        "confidence":      float(row["confidence"]),
        "source_datetime": source_datetime_str,
        "token_count":     token_count_from_text(memory_text),
        "conversation_id": row["conversation_id"],
    }
