"""Tests for src/harness/inject.py — TDD, Task 4.5."""
import hashlib
import os

import pytest

from src.harness.inject import stage_memory, verify_injection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------------------
# stage_memory
# ---------------------------------------------------------------------------

class TestStageMemory:
    def test_copies_memory_md_byte_identical(self, tmp_path):
        src = tmp_path / "source.md"
        src.write_bytes(b"# MEMORY\n\nSome profile content.\n<!-- SENTINEL:abc123 -->")
        workdir = tmp_path / "work"

        staged = stage_memory(str(src), str(workdir))

        assert staged == str(workdir / "MEMORY.md")
        assert open(staged, "rb").read() == src.read_bytes()

    def test_creates_workdir_if_missing(self, tmp_path):
        src = tmp_path / "source.md"
        src.write_bytes(b"# MEMORY\n")
        workdir = tmp_path / "new" / "nested" / "dir"

        assert not workdir.exists()
        stage_memory(str(src), str(workdir))
        assert workdir.exists()

    def test_writes_claude_md_with_exact_content(self, tmp_path):
        src = tmp_path / "source.md"
        src.write_bytes(b"# MEMORY\n")
        workdir = tmp_path / "work"

        stage_memory(str(src), str(workdir))

        claude_md = workdir / "CLAUDE.md"
        assert claude_md.exists()
        assert claude_md.read_text() == "@MEMORY.md\n"

    def test_staged_path_is_workdir_memory_md(self, tmp_path):
        src = tmp_path / "source.md"
        src.write_bytes(b"content")
        workdir = tmp_path / "work"

        result = stage_memory(str(src), str(workdir))

        assert result == str(workdir / "MEMORY.md")

    def test_header_only_source_no_special_casing(self, tmp_path):
        """Empty arm: a header-only file stages identically — no special-casing."""
        header_only = tmp_path / "empty_arm.md"
        header_only.write_bytes(b"# MEMORY\n<!-- SENTINEL:empty -->\n")
        workdir = tmp_path / "work"

        staged = stage_memory(str(header_only), str(workdir))

        assert open(staged, "rb").read() == header_only.read_bytes()
        assert (workdir / "CLAUDE.md").read_text() == "@MEMORY.md\n"

    def test_idempotent_overwrite(self, tmp_path):
        """Calling stage_memory twice with different source overwrites cleanly."""
        src1 = tmp_path / "v1.md"
        src1.write_bytes(b"version 1")
        src2 = tmp_path / "v2.md"
        src2.write_bytes(b"version 2")
        workdir = tmp_path / "work"

        stage_memory(str(src1), str(workdir))
        stage_memory(str(src2), str(workdir))

        assert open(str(workdir / "MEMORY.md"), "rb").read() == b"version 2"


# ---------------------------------------------------------------------------
# verify_injection
# ---------------------------------------------------------------------------

class TestVerifyInjection:
    def _stage_and_sha(self, tmp_path, content: bytes) -> tuple[str, str]:
        src = tmp_path / "source.md"
        src.write_bytes(content)
        workdir = tmp_path / "work"
        staged = stage_memory(str(src), str(workdir))
        sha = _sha256(staged)
        return staged, sha

    def test_returns_true_when_sentinel_echoed_and_sha_matches(self, tmp_path):
        content = b"# MEMORY\n<!-- SENTINEL:tok99 -->\n"
        staged, sha = self._stage_and_sha(tmp_path, content)
        echo = "Here is my memory: <!-- SENTINEL:tok99 --> some text after"

        assert verify_injection(echo, "<!-- SENTINEL:tok99 -->", staged, sha) is True

    def test_false_when_sentinel_absent(self, tmp_path):
        content = b"# MEMORY\n<!-- SENTINEL:tok99 -->\n"
        staged, sha = self._stage_and_sha(tmp_path, content)
        echo = "No sentinel here at all."

        assert verify_injection(echo, "<!-- SENTINEL:tok99 -->", staged, sha) is False

    def test_false_when_sha_mismatches(self, tmp_path):
        content = b"# MEMORY\n<!-- SENTINEL:tok99 -->\n"
        staged, _ = self._stage_and_sha(tmp_path, content)
        wrong_sha = "a" * 64  # wrong but plausible length

        echo = "<!-- SENTINEL:tok99 -->"
        assert verify_injection(echo, "<!-- SENTINEL:tok99 -->", staged, wrong_sha) is False

    def test_false_when_staged_file_missing(self, tmp_path):
        missing_path = str(tmp_path / "nonexistent" / "MEMORY.md")
        sha = "b" * 64  # doesn't matter — file won't be found

        assert verify_injection("<!-- SENTINEL:x -->", "<!-- SENTINEL:x -->", missing_path, sha) is False

    def test_round_trip_stage_then_verify(self, tmp_path):
        """Full round-trip: stage a real file, compute sha, verify succeeds."""
        content = b"# MEMORY\n\nProfile data.\n<!-- SENTINEL:ROUND_TRIP_42 -->\n"
        sentinel = "<!-- SENTINEL:ROUND_TRIP_42 -->"
        src = tmp_path / "source.md"
        src.write_bytes(content)
        workdir = tmp_path / "work"

        staged = stage_memory(str(src), str(workdir))
        sha = _sha256(staged)
        echo = f"I read the memory file and found {sentinel} in it."

        assert verify_injection(echo, sentinel, staged, sha) is True

    def test_false_when_both_sentinel_absent_and_sha_wrong(self, tmp_path):
        content = b"# MEMORY\n"
        staged, _ = self._stage_and_sha(tmp_path, content)
        assert verify_injection("no sentinel", "<!-- SENTINEL:x -->", staged, "z" * 64) is False
