"""
JSON-schema validation for extracted atomic memories.

Implements the SPEC §3.2 constraints as pure functions that return
human-readable violation messages — no exceptions for normal validation
failures.  No I/O; callers pass data in.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Memory type enum (SPEC §3.2 — 8 values, exact)
# ---------------------------------------------------------------------------

MEMORY_TYPES: frozenset[str] = frozenset({
    "user_preference",
    "workflow_preference",
    "repository_fact",
    "architecture_decision",
    "command_or_environment",
    "failure_and_fix",
    "code_convention",
    "project_state",
})

_MEMORY_TEXT_MIN = 15
_MEMORY_TEXT_MAX = 500
_ARRAY_CAP = 20


# ---------------------------------------------------------------------------
# Single-item validator
# ---------------------------------------------------------------------------

def validate_memory(obj: dict) -> list[str]:
    """Validate a single memory dict against the SPEC §3.2 schema.

    Checks performed:
    - ``memory_type`` ∈ MEMORY_TYPES (the 8-value enum).
    - ``15 ≤ len(memory_text) ≤ 500``.
    - ``0.0 ≤ confidence ≤ 1.0`` (key must be present).
    - ``evidence`` is present, non-empty, and references a chat_step
      (contains at least one digit — chat_step is a BIGINT).

    Args:
        obj: A dict representing one candidate memory object.

    Returns:
        A list of human-readable violation messages.  An empty list means
        the object is valid.
    """
    errors: list[str] = []

    # --- memory_type ---
    mt = obj.get("memory_type")
    if mt not in MEMORY_TYPES:
        errors.append(
            f"memory_type {mt!r} is not in the allowed enum "
            f"{sorted(MEMORY_TYPES)}"
        )

    # --- memory_text length ---
    text = obj.get("memory_text") or ""
    text_len = len(text)
    if not (_MEMORY_TEXT_MIN <= text_len <= _MEMORY_TEXT_MAX):
        errors.append(
            f"memory_text length {text_len} is out of range "
            f"[{_MEMORY_TEXT_MIN}, {_MEMORY_TEXT_MAX}]"
        )

    # --- confidence ---
    if "confidence" not in obj:
        errors.append("confidence is missing")
    else:
        conf = obj["confidence"]
        if conf is None or not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            errors.append(
                f"confidence {conf!r} is out of range [0.0, 1.0]"
            )

    # --- evidence ---
    evidence = obj.get("evidence")
    if not evidence:
        errors.append("evidence is missing or empty")
    else:
        evidence_str = str(evidence)
        if not any(c.isdigit() for c in evidence_str):
            errors.append(
                "evidence does not reference a chat_step "
                "(expected at least one digit, e.g. 'step 4: …')"
            )

    return errors


# ---------------------------------------------------------------------------
# Array validator
# ---------------------------------------------------------------------------

def validate_memory_array(arr: list[dict]) -> list[str]:
    """Validate an array of memory objects.

    Enforces the ≤20-item cap and aggregates per-item errors.

    Args:
        arr: List of memory dicts.

    Returns:
        A list of human-readable violation messages.  Empty = fully valid.
    """
    errors: list[str] = []

    if len(arr) > _ARRAY_CAP:
        errors.append(
            f"memory array exceeds the {_ARRAY_CAP}-item cap: "
            f"got {len(arr)} items"
        )
        # Still validate what's there so callers get full feedback.

    for i, item in enumerate(arr):
        item_errors = validate_memory(item)
        for err in item_errors:
            errors.append(f"item[{i}]: {err}")

    return errors
