"""
compiler.render — deterministic MEMORY.md render (v1) + sentinel + hashes.

Public surface (Task 4.3):
    render(task_id, arm, mems, cfg) -> RenderResult

RenderResult is a NamedTuple with fields:
    markdown      : str    — the fully rendered MEMORY.md text
    file_sha256   : str    — sha256 hexdigest of markdown.encode("utf-8")
    payload_sha256: str    — sha256 of the canonical mems+cfg payload (no circularity)
    sentinel      : str    — "mem-{task_id}-{arm}-{shortsha}"
    token_count   : int    — whitespace-separated token count of markdown

Sentinel/hash ordering (avoids circularity):
    1. payload_sha256 — sha256 of canonical mems+cfg string; does NOT touch markdown.
    2. shortsha       — first 8 hex chars of payload_sha256.
    3. sentinel       — f"mem-{task_id}-{arm}-{shortsha}".
    4. markdown       — v1 template rendered with sentinel embedded.
    5. file_sha256    — sha256(markdown.encode("utf-8")).
"""

from __future__ import annotations

import hashlib
import re
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Section order and memory_type → section mapping (verbatim from SPEC §5 v1)
# ---------------------------------------------------------------------------

# Each entry: (section_header, tuple of memory_types that map to this section)
# Order here defines the order sections appear in the rendered markdown.
_SECTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("## User & Working Style",         ("user_preference", "workflow_preference")),
    ("## Commands & Environment",       ("command_or_environment",)),
    ("## Project / Repo Facts",         ("repository_fact",)),
    ("## Decisions & Rationale",        ("architecture_decision",)),
    ("## Known Pitfalls & Fixes",       ("failure_and_fix",)),
    ("## Conventions",                  ("code_convention",)),
    ("## In-Flight / Superseded State", ("project_state",)),
]

# The fixed header/how-to-use block.  {sentinel} is a single format placeholder.
_HEADER_TEMPLATE = (
    "<!-- MEMORY_SENTINEL: {sentinel} -->\n"
    "# MEMORY.md\n"
    "## How to use this file\n"
    "Prior context about the user (Abdullah) and their projects, distilled from past sessions.\n"
    "Treat as trusted background. Prefer its commands, conventions, and file locations over\n"
    "re-exploring. Do not restate it. Before acting, acknowledge memory version {sentinel}."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Strip, collapse internal whitespace, lowercase.

    Used for the canonical payload string — NOT applied to rendered markdown.
    """
    return re.sub(r"\s+", " ", text.strip()).lower()


def _compute_payload_sha256(mems: list[dict], cfg: dict) -> str:
    """SHA-256 hexdigest of the canonical payload string.

    Canonical form (does NOT involve the rendered markdown):
        One line per mem: "{memory_id}|{norm_text}|{memory_type}"
        Followed by the render_template_version line.
        Lines joined by newline, encoded UTF-8.

    This must be computed BEFORE sentinel/markdown so there is no circularity.
    """
    lines = [
        f"{m['memory_id']}|{_normalise(m['memory_text'])}|{m['memory_type']}"
        for m in mems
    ]
    lines.append(cfg["render_template_version"])
    canonical = "\n".join(lines)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RenderResult(NamedTuple):
    """Return value of render()."""
    markdown: str
    file_sha256: str
    payload_sha256: str
    sentinel: str
    token_count: int


def render(task_id: str, arm: str, mems: list[dict], cfg: dict) -> RenderResult:
    """Render a deterministic MEMORY.md for the given (task_id, arm, mems, cfg).

    Args:
        task_id: Eval task identifier (e.g. "T1").
        arm:     Experimental arm label (e.g. "empty", "retrieved", "placebo").
        mems:    Selected+ranked memory records in the order they should appear.
                 Each record must have: memory_id (str), memory_text (str),
                 memory_type (str — one of the eight SPEC enum values).
                 The caller (arms.pool()) is responsible for selection and ordering;
                 render() preserves the given order within each section.
        cfg:     Compiler config dict.  Must include "render_template_version" key.

    Returns:
        RenderResult NamedTuple (markdown, file_sha256, payload_sha256, sentinel,
        token_count).  Identical inputs always produce byte-identical markdown and
        identical hashes (no randomness, no timestamps, no dict-order dependence).
    """
    # ------------------------------------------------------------------ #
    # Step 1: payload_sha256 — computed from mems + cfg; no markdown yet. #
    # ------------------------------------------------------------------ #
    p_sha = _compute_payload_sha256(mems, cfg)

    # ------------------------------------------------------------------ #
    # Step 2: shortsha + sentinel                                         #
    # ------------------------------------------------------------------ #
    shortsha = p_sha[:8]
    sentinel = f"mem-{task_id}-{arm}-{shortsha}"

    # ------------------------------------------------------------------ #
    # Step 3: render markdown                                             #
    # ------------------------------------------------------------------ #
    # Start with the fixed header block (sentinel embedded here).
    parts: list[str] = [_HEADER_TEMPLATE.format(sentinel=sentinel)]

    # For each SPEC section (in spec order), collect mems whose memory_type
    # belongs to that section, preserving their original input order.
    for section_header, types in _SECTIONS:
        type_set = frozenset(types)
        section_mems = [m for m in mems if m["memory_type"] in type_set]
        if not section_mems:
            continue  # omit empty sections entirely
        lines = [section_header]
        for m in section_mems:
            lines.append(f"- {m['memory_text']}")
        parts.append("\n".join(lines))

    # Join all parts with a single blank line between them.
    markdown = "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Step 4: file_sha256 — computed last, after sentinel is in markdown. #
    # ------------------------------------------------------------------ #
    f_sha = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    # Step 5: token_count — whitespace-separated tokens in markdown.      #
    # ------------------------------------------------------------------ #
    token_count = len(markdown.split())

    return RenderResult(
        markdown=markdown,
        file_sha256=f_sha,
        payload_sha256=p_sha,
        sentinel=sentinel,
        token_count=token_count,
    )
