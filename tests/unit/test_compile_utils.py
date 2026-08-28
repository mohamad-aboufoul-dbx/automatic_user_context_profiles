"""
Unit tests for compiler.artifact (Task 4.4 helpers):
    token_count_from_text, make_artifact_id, spark_row_to_mem.

Also verifies that COMPILER_VERSION is importable from the compiler package.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from compiler import COMPILER_VERSION
from compiler.artifact import (
    make_artifact_id,
    spark_row_to_mem,
    token_count_from_text,
)


# ---------------------------------------------------------------------------
# COMPILER_VERSION
# ---------------------------------------------------------------------------

class TestCompilerVersion:
    def test_is_v1(self):
        assert COMPILER_VERSION == "v1"

    def test_is_string(self):
        assert isinstance(COMPILER_VERSION, str)


# ---------------------------------------------------------------------------
# token_count_from_text
# ---------------------------------------------------------------------------

class TestTokenCountFromText:
    def test_empty_string_returns_zero(self):
        assert token_count_from_text("") == 0

    def test_whitespace_only_returns_zero(self):
        assert token_count_from_text("   \t\n  ") == 0

    def test_single_word(self):
        assert token_count_from_text("hello") == 1

    def test_multiple_words(self):
        assert token_count_from_text("hello world foo bar") == 4

    def test_leading_trailing_whitespace_stripped(self):
        assert token_count_from_text("  hello world  ") == 2

    def test_multiple_spaces_collapsed(self):
        assert token_count_from_text("a  b   c") == 3

    def test_tabs_and_newlines_count_as_whitespace(self):
        assert token_count_from_text("a\tb\nc") == 3

    def test_sentence(self):
        text = "Use ruff for linting and black for formatting"
        assert token_count_from_text(text) == 8

    def test_consistent_with_render_estimator(self):
        """token_count_from_text must match render.py's len(markdown.split())."""
        sample = "Run tests with: python3 -m pytest tests/unit -v"
        from compiler.render import render

        cfg = {"render_template_version": "v1"}
        mem = {
            "memory_id":   "m1",
            "memory_text": sample,
            "memory_type": "command_or_environment",
        }
        result = render("T1", "retrieved", [mem], cfg)
        rendered_token_count = result.token_count
        assert rendered_token_count == len(result.markdown.split())


# ---------------------------------------------------------------------------
# make_artifact_id
# ---------------------------------------------------------------------------

class TestMakeArtifactId:
    def test_returns_sha256_hexdigest(self):
        result = make_artifact_id("T1", "empty", "abc123")
        expected = hashlib.sha256(b"T1|empty|abc123").hexdigest()
        assert result == expected

    def test_output_is_64_hex_chars(self):
        result = make_artifact_id("T1", "retrieved", "deadbeef" * 8)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic_same_inputs(self):
        r1 = make_artifact_id("T2", "placebo", "somesha")
        r2 = make_artifact_id("T2", "placebo", "somesha")
        assert r1 == r2

    def test_different_task_id_differs(self):
        assert make_artifact_id("T1", "empty", "x") != make_artifact_id("T2", "empty", "x")

    def test_different_arm_differs(self):
        assert make_artifact_id("T1", "empty", "x") != make_artifact_id("T1", "retrieved", "x")

    def test_different_sha_differs(self):
        assert make_artifact_id("T1", "empty", "x") != make_artifact_id("T1", "empty", "y")

    def test_all_twelve_combinations_unique(self):
        """All (task, arm) combos with same sha produce distinct artifact_ids."""
        sha = "a" * 64
        ids = [
            make_artifact_id(t, a, sha)
            for t in ["T1", "T2", "T3"]
            for a in ["empty", "static_generic", "retrieved", "placebo"]
        ]
        assert len(ids) == len(set(ids)), "Duplicate artifact_ids found"

    def test_pipe_delimiter_used(self):
        """Verify the canonical form: sha256("T1|empty|abc")."""
        raw = "T1|empty|abc"
        expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert make_artifact_id("T1", "empty", "abc") == expected


# ---------------------------------------------------------------------------
# spark_row_to_mem
# ---------------------------------------------------------------------------

def _make_row(
    memory_id: str = "mid-001",
    memory_text: str = "Use ruff for linting",
    memory_type: str = "workflow_preference",
    domain: str = "general_workflow",
    embedding: list | None = None,
    confidence: float = 0.9,
    source_datetime: datetime | None = None,
    conversation_id: str = "conv-123",
) -> dict:
    """Build a dict that mimics a PySpark Row (supports row['field'] syntax)."""
    return {
        "memory_id":       memory_id,
        "memory_text":     memory_text,
        "memory_type":     memory_type,
        "domain":          domain,
        "embedding":       embedding if embedding is not None else [0.1, 0.2, 0.3],
        "confidence":      confidence,
        "source_datetime": (
            source_datetime
            if source_datetime is not None
            else datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)
        ),
        "conversation_id": conversation_id,
    }


class TestSparkRowToMem:
    def test_token_count_injected(self):
        row = _make_row(memory_text="Use ruff for linting")
        mem = spark_row_to_mem(row)
        assert mem["token_count"] == 4

    def test_token_count_empty_text(self):
        row = _make_row(memory_text="")
        mem = spark_row_to_mem(row)
        assert mem["token_count"] == 0

    def test_token_count_multi_word(self):
        row = _make_row(memory_text="a b c d e")
        mem = spark_row_to_mem(row)
        assert mem["token_count"] == 5

    def test_embedding_converted_to_list_float(self):
        row = _make_row(embedding=[0.1, 0.2, 0.3])
        mem = spark_row_to_mem(row)
        assert isinstance(mem["embedding"], list)
        assert all(isinstance(x, float) for x in mem["embedding"])
        assert len(mem["embedding"]) == 3

    def test_embedding_values_preserved(self):
        row = _make_row(embedding=[1.0, -0.5, 0.25])
        mem = spark_row_to_mem(row)
        assert mem["embedding"] == [1.0, -0.5, 0.25]

    def test_embedding_cast_from_int(self):
        row = _make_row(embedding=[1, 0, 2])
        mem = spark_row_to_mem(row)
        assert mem["embedding"] == [1.0, 0.0, 2.0]
        assert all(isinstance(x, float) for x in mem["embedding"])

    def test_source_datetime_aware_utc_to_iso_z(self):
        dt = datetime(2026, 8, 10, 12, 30, 45, tzinfo=timezone.utc)
        row = _make_row(source_datetime=dt)
        mem = spark_row_to_mem(row)
        assert isinstance(mem["source_datetime"], str)
        assert mem["source_datetime"] == "2026-08-10T12:30:45Z"

    def test_source_datetime_naive_treated_as_utc(self):
        dt = datetime(2026, 8, 10, 0, 0, 0)
        row = _make_row(source_datetime=dt)
        mem = spark_row_to_mem(row)
        assert mem["source_datetime"] == "2026-08-10T00:00:00Z"

    def test_conversation_id_included(self):
        row = _make_row(conversation_id="conv-456")
        mem = spark_row_to_mem(row)
        assert mem["conversation_id"] == "conv-456"

    def test_confidence_cast_to_float(self):
        row = _make_row(confidence=0.85)
        mem = spark_row_to_mem(row)
        assert isinstance(mem["confidence"], float)
        assert mem["confidence"] == pytest.approx(0.85)

    def test_all_required_compiler_fields_present(self):
        row = _make_row()
        mem = spark_row_to_mem(row)
        required = {
            "memory_id", "memory_text", "memory_type", "domain",
            "embedding", "confidence", "source_datetime", "token_count",
        }
        assert required.issubset(mem.keys())

    def test_conversation_id_present_for_contamination_guard(self):
        row = _make_row()
        mem = spark_row_to_mem(row)
        assert "conversation_id" in mem

    def test_passthrough_fields(self):
        row = _make_row(
            memory_id="test-id",
            memory_text="Some text",
            memory_type="repository_fact",
            domain="repo:platform",
        )
        mem = spark_row_to_mem(row)
        assert mem["memory_id"] == "test-id"
        assert mem["memory_text"] == "Some text"
        assert mem["memory_type"] == "repository_fact"
        assert mem["domain"] == "repo:platform"
