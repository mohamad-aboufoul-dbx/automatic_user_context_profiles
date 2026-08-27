# Automatic User Context Profiles — Build Checklist

Source of truth: [SPEC.md](SPEC.md). Update this checklist as work lands; the spec and its hard invariants take precedence.

## Repository and Day 1 foundation

- [x] Clone the team repository.
- [x] Create `dev/asaid` from the default branch without modifying `dev/maboufoul`.
- [x] Add the build spec at `docs/SPEC.md`.
- [x] Scaffold `notebooks/`, `config/`, `prompts/`, and `eval/tasks/`.
- [x] Authenticate the Databricks CLI with profile `hackathon`.
- [x] Create the Unity Catalog schema objects from SPEC §3.
- [x] Load `raw_conversations_abdullah_said` using an idempotent `row_hash` MERGE.
- [x] Validate the raw load is non-empty, approximately 247 rows, and contains only `username = "abdullah.said"`.
- [x] Sessionize by `conversation_id`, ordered by `event_datetime` and `chat_step`.
- [x] Complete a live setup → ingest → sessionize run with validated completion sentinels.
- [ ] Complete independent cross-vendor review of PR #2.
- [ ] Merge PR #2 into `dev/asaid` after review passes (human action).

## Held-out evaluation gate — approval required

- [ ] Record the verified raw row count and a representative sample session.
- [ ] Propose `CUTOFF_TS`.
- [ ] Propose three held-out `/goal` tasks with pinned commits.
- [ ] Obtain explicit human approval for the cutoff and held-out task set.
- [ ] Record held-out `conversation_id` values for contamination exclusion.

> **STOP:** Do not begin extraction (`02`) until the cutoff and held-out set are explicitly approved.

## Memory extraction and compilation

- [ ] Apply the temporal cutoff before extraction.
- [ ] Exclude all held-out conversation IDs before extraction.
- [ ] Ensure no held-out task's final solution is extracted.
- [ ] Implement extraction into the shared `atomic_memories` contract from SPEC §3.2.
- [ ] Keep `MEMORY.md` tool-agnostic.
- [ ] Add the one-line Claude Code loader: `CLAUDE.md` → `@MEMORY.md`.
- [ ] Compile one frozen `MEMORY.md` for every `(task, arm)` pair offline.
- [ ] Hash every compiled memory artifact.
- [ ] Reuse each frozen artifact byte-for-byte across repeats.
- [ ] Use the same loader for every arm; vary only file contents.
- [ ] Make the empty arm header-only.

## Offline-frozen evaluation

- [ ] Define and validate evaluation task manifests in `eval/tasks/`.
- [ ] Pin repository commits for all scored tasks.
- [ ] Verify scored runs perform no live retrieval.
- [ ] Run baseline and treatment arms using their frozen memory artifacts.
- [ ] Never hand-edit a treatment file after a run.
- [ ] Measure success before efficiency metrics.
- [ ] Persist run metadata, artifact hashes, outcomes, and efficiency metrics.
- [ ] Produce the clean comparison required by the spec.

## Final verification

- [ ] Re-run ingestion to demonstrate idempotency.
- [ ] Verify table schemas against SPEC §3.
- [ ] Verify contamination guards from stored run evidence.
- [ ] Verify frozen artifact hashes and repeat reuse.
- [ ] Cross-review remaining implementation PRs with a different vendor.
- [ ] Merge approved PRs into `dev/asaid` (human action).
