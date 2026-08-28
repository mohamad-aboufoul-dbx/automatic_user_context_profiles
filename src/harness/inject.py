"""
Tool-agnostic memory loader + injection/sentinel verifier — Task 4.5.

Every experimental arm stages its frozen MEMORY.md the same way via
``stage_memory``; only the file contents differ.  ``verify_injection``
confirms the agent actually received and echoed the sentinel AND that the
staged bytes are bit-for-bit identical to the expected artifact.
"""

import hashlib
import os
import shutil


def stage_memory(md_path: str, workdir: str) -> str:
    """Copy the frozen MEMORY.md at *md_path* into *workdir*.

    Writes:
    - ``<workdir>/MEMORY.md``  — byte-for-byte copy of the source file.
    - ``<workdir>/CLAUDE.md``  — contains exactly ``@MEMORY.md\\n``.

    Creates *workdir* (and any parents) if it does not already exist.

    Returns the path to the staged ``MEMORY.md`` file.
    """
    os.makedirs(workdir, exist_ok=True)

    staged_path = os.path.join(workdir, "MEMORY.md")
    shutil.copy2(md_path, staged_path)

    claude_md_path = os.path.join(workdir, "CLAUDE.md")
    with open(claude_md_path, "w", encoding="utf-8") as fh:
        fh.write("@MEMORY.md\n")

    return staged_path


def verify_injection(
    agent_echo: str,
    expected_sentinel: str,
    staged_path: str,
    expected_sha: str,
) -> bool:
    """Return True iff the agent echoed the sentinel AND the staged file matches.

    Conditions (both must hold):
    1. ``expected_sentinel in agent_echo``
    2. ``sha256(bytes of file at staged_path).hexdigest() == expected_sha``

    Returns False if the sentinel is absent, the sha mismatches, or the file
    at *staged_path* does not exist.
    """
    if expected_sentinel not in agent_echo:
        return False

    try:
        raw = open(staged_path, "rb").read()
    except (FileNotFoundError, OSError):
        return False

    return hashlib.sha256(raw).hexdigest() == expected_sha
