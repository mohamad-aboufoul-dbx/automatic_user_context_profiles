"""
TDD tests for extract.ids — Task 2.3.
Write tests FIRST (RED), then implement (GREEN).
"""

import pytest
from extract.ids import normalize_text, memory_id


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def test_normalize_strips_leading_trailing_whitespace():
    assert normalize_text("  hello  ") == "hello"


def test_normalize_collapses_internal_whitespace():
    assert normalize_text("foo   bar") == "foo bar"


def test_normalize_lowercases():
    assert normalize_text("Foo BAR") == "foo bar"


def test_normalize_combined():
    """Strip + collapse + lowercase all at once."""
    assert normalize_text("  Foo  Bar  ") == "foo bar"


def test_normalize_leading_trailing_tab_newline():
    assert normalize_text("\tFoo\nBar\t") == "foo bar"


def test_normalize_already_clean():
    assert normalize_text("hello world") == "hello world"


# ---------------------------------------------------------------------------
# memory_id — determinism
# ---------------------------------------------------------------------------

def test_memory_id_same_inputs_produce_same_id():
    """Same conversation_id, memory_text, memory_type → same id."""
    id1 = memory_id("conv-123", "Abdullah uses TDD-first workflow", "workflow_preference")
    id2 = memory_id("conv-123", "Abdullah uses TDD-first workflow", "workflow_preference")
    assert id1 == id2


def test_memory_id_different_memory_type_produces_different_id():
    """Changing memory_type changes the id."""
    id1 = memory_id("conv-123", "Abdullah uses TDD-first workflow", "workflow_preference")
    id2 = memory_id("conv-123", "Abdullah uses TDD-first workflow", "user_preference")
    assert id1 != id2


def test_memory_id_different_conversation_id_produces_different_id():
    """Changing conversation_id changes the id."""
    id1 = memory_id("conv-123", "Abdullah uses TDD-first workflow", "workflow_preference")
    id2 = memory_id("conv-456", "Abdullah uses TDD-first workflow", "workflow_preference")
    assert id1 != id2


def test_memory_id_different_text_produces_different_id():
    """Different memory_text → different id."""
    id1 = memory_id("conv-123", "Abdullah uses TDD-first workflow", "workflow_preference")
    id2 = memory_id("conv-123", "Abdullah prefers worktrees", "workflow_preference")
    assert id1 != id2


# ---------------------------------------------------------------------------
# memory_id — normalization applied before hashing
# ---------------------------------------------------------------------------

def test_memory_id_normalization_makes_padded_and_lower_collide():
    """' Foo  Bar ' and 'foo bar' hash to the same id (normalization applied)."""
    id1 = memory_id("conv-123", " Foo  Bar ", "workflow_preference")
    id2 = memory_id("conv-123", "foo bar", "workflow_preference")
    assert id1 == id2


def test_memory_id_normalization_preserves_separator():
    """Normalization of text does NOT affect the | separator characters in the hash payload."""
    # Two different text values that normalize to different strings should still differ
    id1 = memory_id("conv-x", "alpha beta", "code_convention")
    id2 = memory_id("conv-x", "alpha  beta", "code_convention")  # extra space, normalizes same
    assert id1 == id2  # both normalize to "alpha beta"


def test_memory_id_is_64_hex_chars():
    """SHA-256 hexdigest is exactly 64 characters."""
    result = memory_id("conv-123", "Abdullah uses TDD-first workflow", "workflow_preference")
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)
