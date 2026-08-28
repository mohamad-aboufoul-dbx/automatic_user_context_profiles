"""
TDD tests for extract.schema — Task 2.2.
Write tests FIRST (RED), then implement (GREEN).
"""

import pytest
from extract.schema import MEMORY_TYPES, validate_memory, validate_memory_array


# ---------------------------------------------------------------------------
# Canonical test from the brief
# ---------------------------------------------------------------------------

def test_memory_schema_enforces_enum_and_length():
    """From task-2.2-brief: enum rejection + short text + confidence >1."""
    bad = {
        "memory_text": "x",
        "memory_type": "not_an_enum",
        "domain": "repo:platform",
        "evidence": "step 4: user ran pytest and it passed",
        "confidence": 1.5,
    }
    errs = validate_memory(bad)
    assert any("memory_type" in e for e in errs)
    assert any("memory_text" in e for e in errs)   # <15 chars
    assert any("confidence" in e for e in errs)     # >1


# ---------------------------------------------------------------------------
# MEMORY_TYPES enum completeness
# ---------------------------------------------------------------------------

def test_memory_types_has_exactly_8_values():
    """SPEC §3.2 defines exactly 8 memory_type values."""
    assert len(MEMORY_TYPES) == 8


def test_memory_types_contains_all_spec_values():
    expected = {
        "user_preference",
        "workflow_preference",
        "repository_fact",
        "architecture_decision",
        "command_or_environment",
        "failure_and_fix",
        "code_convention",
        "project_state",
    }
    assert MEMORY_TYPES == expected


# ---------------------------------------------------------------------------
# memory_type validation
# ---------------------------------------------------------------------------

def test_valid_memory_type_passes():
    obj = {
        "memory_text": "Abdullah uses TDD-first workflow for all features",
        "memory_type": "workflow_preference",
        "evidence": "step 3: user wrote tests before implementation",
        "confidence": 0.9,
    }
    assert validate_memory(obj) == []


def test_invalid_memory_type_rejected():
    obj = {
        "memory_text": "Abdullah uses TDD-first workflow for all features",
        "memory_type": "made_up_type",
        "evidence": "step 3: user wrote tests before implementation",
        "confidence": 0.9,
    }
    errs = validate_memory(obj)
    assert any("memory_type" in e for e in errs)


def test_none_memory_type_rejected():
    obj = {
        "memory_text": "Abdullah uses TDD-first workflow for all features",
        "memory_type": None,
        "evidence": "step 3: example",
        "confidence": 0.5,
    }
    errs = validate_memory(obj)
    assert any("memory_type" in e for e in errs)


# ---------------------------------------------------------------------------
# memory_text length bounds
# ---------------------------------------------------------------------------

def test_memory_text_exactly_15_chars_is_valid():
    obj = {
        "memory_text": "a" * 15,
        "memory_type": "code_convention",
        "evidence": "step 1: confirmed",
        "confidence": 0.5,
    }
    assert validate_memory(obj) == []


def test_memory_text_14_chars_is_invalid():
    """14 characters is below the minimum of 15."""
    obj = {
        "memory_text": "a" * 14,
        "memory_type": "code_convention",
        "evidence": "step 1: confirmed",
        "confidence": 0.5,
    }
    errs = validate_memory(obj)
    assert any("memory_text" in e for e in errs)


def test_memory_text_exactly_500_chars_is_valid():
    obj = {
        "memory_text": "a" * 500,
        "memory_type": "repository_fact",
        "evidence": "step 2: confirmed in tool output",
        "confidence": 0.7,
    }
    assert validate_memory(obj) == []


def test_memory_text_501_chars_is_invalid():
    """501 characters exceeds the maximum of 500."""
    obj = {
        "memory_text": "a" * 501,
        "memory_type": "repository_fact",
        "evidence": "step 2: confirmed",
        "confidence": 0.7,
    }
    errs = validate_memory(obj)
    assert any("memory_text" in e for e in errs)


# ---------------------------------------------------------------------------
# confidence bounds
# ---------------------------------------------------------------------------

def test_confidence_zero_is_valid():
    obj = {
        "memory_text": "Abdullah uses TDD-first workflow",
        "memory_type": "workflow_preference",
        "evidence": "step 1: noted",
        "confidence": 0.0,
    }
    assert validate_memory(obj) == []


def test_confidence_one_is_valid():
    obj = {
        "memory_text": "Abdullah uses TDD-first workflow",
        "memory_type": "workflow_preference",
        "evidence": "step 1: noted",
        "confidence": 1.0,
    }
    assert validate_memory(obj) == []


def test_confidence_negative_is_invalid():
    obj = {
        "memory_text": "Abdullah uses TDD-first workflow",
        "memory_type": "workflow_preference",
        "evidence": "step 1: noted",
        "confidence": -0.1,
    }
    errs = validate_memory(obj)
    assert any("confidence" in e for e in errs)


def test_confidence_greater_than_one_is_invalid():
    obj = {
        "memory_text": "Abdullah uses TDD-first workflow",
        "memory_type": "workflow_preference",
        "evidence": "step 1: noted",
        "confidence": 1.01,
    }
    errs = validate_memory(obj)
    assert any("confidence" in e for e in errs)


def test_confidence_missing_is_invalid():
    obj = {
        "memory_text": "Abdullah uses TDD-first workflow",
        "memory_type": "workflow_preference",
        "evidence": "step 1: noted",
    }
    errs = validate_memory(obj)
    assert any("confidence" in e for e in errs)


# ---------------------------------------------------------------------------
# evidence validation
# ---------------------------------------------------------------------------

def test_evidence_missing_is_invalid():
    obj = {
        "memory_text": "Abdullah uses TDD-first workflow",
        "memory_type": "workflow_preference",
        "confidence": 0.8,
    }
    errs = validate_memory(obj)
    assert any("evidence" in e for e in errs)


def test_evidence_empty_string_is_invalid():
    obj = {
        "memory_text": "Abdullah uses TDD-first workflow",
        "memory_type": "workflow_preference",
        "evidence": "",
        "confidence": 0.8,
    }
    errs = validate_memory(obj)
    assert any("evidence" in e for e in errs)


def test_evidence_none_is_invalid():
    obj = {
        "memory_text": "Abdullah uses TDD-first workflow",
        "memory_type": "workflow_preference",
        "evidence": None,
        "confidence": 0.8,
    }
    errs = validate_memory(obj)
    assert any("evidence" in e for e in errs)


# ---------------------------------------------------------------------------
# validate_memory_array — ≤20 cap and aggregated errors
# ---------------------------------------------------------------------------

def _valid_memory(suffix: str = "") -> dict:
    return {
        "memory_text": f"Abdullah uses TDD-first workflow always{suffix}",
        "memory_type": "workflow_preference",
        "evidence": "step 5: confirmed in transcript",
        "confidence": 0.8,
    }


def test_array_within_20_is_valid():
    arr = [_valid_memory(str(i)) for i in range(20)]
    assert validate_memory_array(arr) == []


def test_array_of_21_is_invalid():
    """Array with 21 items exceeds the ≤20 cap."""
    arr = [_valid_memory(str(i)) for i in range(21)]
    errs = validate_memory_array(arr)
    assert len(errs) > 0
    assert any("20" in e or "exceed" in e.lower() or "cap" in e.lower() or "array" in e.lower() for e in errs)


def test_array_aggregates_per_item_errors():
    """Invalid items in array produce per-item error messages."""
    arr = [
        _valid_memory("0"),
        {
            "memory_text": "x",           # too short
            "memory_type": "bad_type",    # invalid enum
            "evidence": "step 1: noted",
            "confidence": 2.0,            # >1
        },
        _valid_memory("2"),
    ]
    errs = validate_memory_array(arr)
    assert len(errs) > 0


def test_array_empty_is_valid():
    assert validate_memory_array([]) == []
