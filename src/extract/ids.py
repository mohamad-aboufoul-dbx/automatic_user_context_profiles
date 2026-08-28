"""
Deterministic memory identity functions.

No I/O; no third-party dependencies (stdlib only).
"""

from __future__ import annotations

import hashlib
import re


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    """Normalise a string for stable hashing.

    Steps applied in order:
    1. Strip leading and trailing whitespace (including tabs, newlines).
    2. Collapse any run of internal whitespace to a single space.
    3. Lowercase.

    Examples::

        normalize_text("  Foo  Bar  ")  # → "foo bar"
        normalize_text("\\tHello\\nWorld\\t")  # → "hello world"
    """
    return re.sub(r"\s+", " ", s.strip()).lower()


# ---------------------------------------------------------------------------
# Deterministic memory id
# ---------------------------------------------------------------------------

def memory_id(conversation_id: str, memory_text: str, memory_type: str) -> str:
    """Return a deterministic SHA-256 hex digest identifying a memory.

    Formula::

        sha256(f"{conversation_id}|{normalize_text(memory_text)}|{memory_type}").hexdigest()

    ``memory_text`` is normalised before hashing so that surface differences
    in whitespace or casing (e.g. ``" Foo  Bar "`` vs ``"foo bar"``) produce
    the same id.

    Args:
        conversation_id: The source conversation's id string.
        memory_text:     The raw memory text (normalisation applied internally).
        memory_type:     One of the 8 MEMORY_TYPES values.

    Returns:
        A 64-character lowercase hex string (SHA-256 digest).
    """
    normalised = normalize_text(memory_text)
    payload = f"{conversation_id}|{normalised}|{memory_type}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
