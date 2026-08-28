"""
Unit tests for compiler.render (Task 4.3).

TDD: written before the implementation.

Coverage:
  - Byte-determinism: same (task_id, arm, mems, cfg) → identical file_sha256
  - Sentinel shape: matches mem-{task_id}-{arm}-<8 hex> and is present in markdown
  - Empty arm: sentinel present, no bullets, no section headers
  - Omitted section: memory_type with no members → section absent from markdown
  - Input-order preservation: bullets appear in mems input order within each section
"""

from __future__ import annotations

import re

import pytest

from compiler.render import RenderResult, render

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CFG = {"render_template_version": "v1"}

MEM_PREF = {
    "memory_id": "m1",
    "memory_text": "Prefers TDD-first workflow",
    "memory_type": "user_preference",
    "domain": "general",
    "confidence": 0.9,
    "source_datetime": "2026-08-01T00:00:00Z",
    "token_count": 4,
    "embedding": [0.1, 0.2],
}

MEM_WORKFLOW = {
    "memory_id": "m3",
    "memory_text": "Uses git worktrees for parallel branch work",
    "memory_type": "workflow_preference",
    "domain": "general",
    "confidence": 0.85,
    "source_datetime": "2026-08-02T00:00:00Z",
    "token_count": 8,
    "embedding": [0.15, 0.25],
}

MEM_CMD = {
    "memory_id": "m2",
    "memory_text": "Run tests with: python3 -m pytest tests/unit -v",
    "memory_type": "command_or_environment",
    "domain": "repo:memory",
    "confidence": 0.95,
    "source_datetime": "2026-08-03T00:00:00Z",
    "token_count": 9,
    "embedding": [0.3, 0.4],
}

MEM_REPO = {
    "memory_id": "m4",
    "memory_text": "Source layout: src/compiler/ holds score.py, arms.py, render.py",
    "memory_type": "repository_fact",
    "domain": "repo:memory",
    "confidence": 0.9,
    "source_datetime": "2026-08-04T00:00:00Z",
    "token_count": 12,
    "embedding": [0.5, 0.6],
}

MEMS = [MEM_PREF, MEM_CMD]


# ---------------------------------------------------------------------------
# Test: return type is RenderResult NamedTuple
# ---------------------------------------------------------------------------

def test_render_returns_render_result():
    out = render("T1", "retrieved", MEMS, CFG)
    assert isinstance(out, RenderResult)
    assert hasattr(out, "markdown")
    assert hasattr(out, "file_sha256")
    assert hasattr(out, "payload_sha256")
    assert hasattr(out, "sentinel")
    assert hasattr(out, "token_count")


# ---------------------------------------------------------------------------
# Test: byte-determinism (the brief's primary test)
# ---------------------------------------------------------------------------

def test_render_is_byte_deterministic():
    out1 = render("T1", "retrieved", MEMS, CFG)
    out2 = render("T1", "retrieved", MEMS, CFG)
    assert out1.file_sha256 == out2.file_sha256
    assert out1.sentinel.startswith("mem-T1-retrieved-")


def test_render_deterministic_payload_sha256():
    out1 = render("T1", "retrieved", MEMS, CFG)
    out2 = render("T1", "retrieved", MEMS, CFG)
    assert out1.payload_sha256 == out2.payload_sha256


def test_render_different_inputs_different_hashes():
    out_a = render("T1", "retrieved", MEMS, CFG)
    out_b = render("T1", "retrieved", [MEM_PREF], CFG)
    assert out_a.file_sha256 != out_b.file_sha256
    assert out_a.payload_sha256 != out_b.payload_sha256


# ---------------------------------------------------------------------------
# Test: sentinel shape and presence in markdown
# ---------------------------------------------------------------------------

_SENTINEL_RE = re.compile(r"^mem-[^-]+-[^-]+-[0-9a-f]{8}$")


def test_sentinel_format():
    out = render("T1", "retrieved", MEMS, CFG)
    assert _SENTINEL_RE.match(out.sentinel), f"Sentinel {out.sentinel!r} does not match pattern"


def test_sentinel_present_in_markdown():
    out = render("T1", "retrieved", MEMS, CFG)
    assert out.sentinel in out.markdown


def test_sentinel_on_comment_line():
    out = render("T1", "retrieved", MEMS, CFG)
    assert f"<!-- MEMORY_SENTINEL: {out.sentinel} -->" in out.markdown


def test_sentinel_in_acknowledge_line():
    out = render("T1", "retrieved", MEMS, CFG)
    assert f"acknowledge memory version {out.sentinel}" in out.markdown


def test_sentinel_encodes_task_and_arm():
    out = render("T2", "placebo", MEMS, CFG)
    assert out.sentinel.startswith("mem-T2-placebo-")


def test_shortsha_is_first_8_chars_of_payload_sha256():
    out = render("T1", "retrieved", MEMS, CFG)
    expected_shortsha = out.payload_sha256[:8]
    assert out.sentinel.endswith(expected_shortsha)


# ---------------------------------------------------------------------------
# Test: payload_sha256 does NOT depend on markdown (anti-circularity)
# ---------------------------------------------------------------------------

def test_payload_sha256_independent_of_sentinel():
    """payload_sha256 is computed before the sentinel; changing task_id/arm
    changes the sentinel but NOT the payload_sha256 (same mems + cfg)."""
    out_a = render("T1", "retrieved", MEMS, CFG)
    out_b = render("T9", "empty_arm_name", MEMS, CFG)
    # Same mems + same cfg → same payload_sha256
    assert out_a.payload_sha256 == out_b.payload_sha256
    # But different sentinel (different task_id/arm)
    assert out_a.sentinel != out_b.sentinel


# ---------------------------------------------------------------------------
# Test: empty arm
# ---------------------------------------------------------------------------

def test_empty_arm_no_mems_has_sentinel():
    out = render("T1", "empty", [], CFG)
    assert out.sentinel in out.markdown
    assert out.sentinel.startswith("mem-T1-empty-")


def test_empty_arm_no_bullets():
    out = render("T1", "empty", [], CFG)
    lines = out.markdown.splitlines()
    bullet_lines = [l for l in lines if l.startswith("- ")]
    assert bullet_lines == [], f"Expected no bullets in empty arm; found: {bullet_lines}"


def test_empty_arm_no_section_headers():
    out = render("T1", "empty", [], CFG)
    assert "## User & Working Style" not in out.markdown
    assert "## Commands & Environment" not in out.markdown
    assert "## Project / Repo Facts" not in out.markdown
    assert "## Decisions & Rationale" not in out.markdown
    assert "## Known Pitfalls & Fixes" not in out.markdown
    assert "## Conventions" not in out.markdown
    assert "## In-Flight / Superseded State" not in out.markdown


def test_empty_arm_still_has_header_block():
    out = render("T1", "empty", [], CFG)
    assert "# MEMORY.md" in out.markdown
    assert "## How to use this file" in out.markdown


def test_empty_arm_is_deterministic():
    out1 = render("T1", "empty", [], CFG)
    out2 = render("T1", "empty", [], CFG)
    assert out1.file_sha256 == out2.file_sha256
    assert out1.markdown == out2.markdown


def test_empty_arm_has_valid_hashes():
    out = render("T1", "empty", [], CFG)
    assert len(out.file_sha256) == 64
    assert all(c in "0123456789abcdef" for c in out.file_sha256)
    assert len(out.payload_sha256) == 64
    assert all(c in "0123456789abcdef" for c in out.payload_sha256)


# ---------------------------------------------------------------------------
# Test: sections omitted when no memories of that type
# ---------------------------------------------------------------------------

def test_missing_type_omits_section():
    """Only user_preference mem → Commands & Environment section is absent."""
    out = render("T1", "retrieved", [MEM_PREF], CFG)
    assert "## User & Working Style" in out.markdown
    assert "## Commands & Environment" not in out.markdown


def test_all_sections_present_when_all_types_given():
    all_mems = [
        {**MEM_PREF, "memory_id": "a1", "memory_type": "user_preference"},
        {**MEM_PREF, "memory_id": "a2", "memory_type": "workflow_preference"},
        {**MEM_PREF, "memory_id": "a3", "memory_type": "command_or_environment"},
        {**MEM_PREF, "memory_id": "a4", "memory_type": "repository_fact"},
        {**MEM_PREF, "memory_id": "a5", "memory_type": "architecture_decision"},
        {**MEM_PREF, "memory_id": "a6", "memory_type": "failure_and_fix"},
        {**MEM_PREF, "memory_id": "a7", "memory_type": "code_convention"},
        {**MEM_PREF, "memory_id": "a8", "memory_type": "project_state"},
    ]
    out = render("T1", "retrieved", all_mems, CFG)
    assert "## User & Working Style" in out.markdown
    assert "## Commands & Environment" in out.markdown
    assert "## Project / Repo Facts" in out.markdown
    assert "## Decisions & Rationale" in out.markdown
    assert "## Known Pitfalls & Fixes" in out.markdown
    assert "## Conventions" in out.markdown
    assert "## In-Flight / Superseded State" in out.markdown


# ---------------------------------------------------------------------------
# Test: input-order preservation
# ---------------------------------------------------------------------------

def test_section_preserves_input_order_same_type():
    """Two mems of same type appear in input order."""
    mem_pref2 = {**MEM_PREF, "memory_id": "m_pref2", "memory_text": "Second pref text"}
    mems = [MEM_PREF, mem_pref2]
    out = render("T1", "retrieved", mems, CFG)
    idx1 = out.markdown.index(MEM_PREF["memory_text"])
    idx2 = out.markdown.index(mem_pref2["memory_text"])
    assert idx1 < idx2


def test_section_preserves_input_order_mixed_types():
    """user_preference and workflow_preference both go to 'User & Working Style'.
    Bullets appear in input order, not grouped by type."""
    # Input order: MEM_PREF (user_pref), MEM_WORKFLOW (wf_pref), MEM_CMD (cmd)
    mems = [MEM_PREF, MEM_WORKFLOW, MEM_CMD]
    out = render("T1", "retrieved", mems, CFG)

    idx_pref = out.markdown.index(MEM_PREF["memory_text"])
    idx_wf = out.markdown.index(MEM_WORKFLOW["memory_text"])
    idx_cmd = out.markdown.index(MEM_CMD["memory_text"])

    # Both user_pref + wf_pref are in User & Working Style, in input order
    assert idx_pref < idx_wf
    # CMD is in a separate section that follows User & Working Style
    assert idx_wf < idx_cmd


def test_multiple_sections_in_spec_order():
    """Sections appear in SPEC order regardless of input order of mems."""
    # Input order reversed: cmd first, then pref
    mems = [MEM_CMD, MEM_PREF]
    out = render("T1", "retrieved", mems, CFG)

    # User & Working Style should come before Commands & Environment
    # because that's the SPEC section order
    idx_user_section = out.markdown.index("## User & Working Style")
    idx_cmd_section = out.markdown.index("## Commands & Environment")
    assert idx_user_section < idx_cmd_section


# ---------------------------------------------------------------------------
# Test: token_count
# ---------------------------------------------------------------------------

def test_token_count_is_whitespace_split():
    out = render("T1", "retrieved", MEMS, CFG)
    expected = len(out.markdown.split())
    assert out.token_count == expected


def test_token_count_empty_arm():
    out = render("T1", "empty", [], CFG)
    expected = len(out.markdown.split())
    assert out.token_count == expected


def test_token_count_positive():
    out = render("T1", "retrieved", MEMS, CFG)
    assert out.token_count > 0


# ---------------------------------------------------------------------------
# Test: file_sha256 is sha256 of utf-8 encoded markdown
# ---------------------------------------------------------------------------

def test_file_sha256_matches_markdown():
    import hashlib
    out = render("T1", "retrieved", MEMS, CFG)
    expected = hashlib.sha256(out.markdown.encode("utf-8")).hexdigest()
    assert out.file_sha256 == expected


# ---------------------------------------------------------------------------
# Test: "How to use this file" block content
# ---------------------------------------------------------------------------

def test_how_to_use_block_present():
    out = render("T1", "retrieved", MEMS, CFG)
    assert "Prior context about the user (Abdullah)" in out.markdown
    assert "Treat as trusted background." in out.markdown
    assert "Do not restate it." in out.markdown


def test_memory_bullets_are_dash_prefixed():
    out = render("T1", "retrieved", MEMS, CFG)
    # Each memory_text should appear as a bullet
    for mem in MEMS:
        assert f"- {mem['memory_text']}" in out.markdown
