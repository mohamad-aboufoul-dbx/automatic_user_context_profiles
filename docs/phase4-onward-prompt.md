# AUCP offline-eval — continuation prompt (Task 4.4 → Phase 7)

> **Standalone prompt.** Hand this to a fresh Claude Code session opened at the repo root
> on branch `dev/asaid` (or a feature branch off it). It carries everything needed to
> finish the Automatic User Context Profiles offline retrieval eval. Read
> `docs/SPEC.md` (source of truth) and `docs/superpowers/plans/2026-08-27-aucp-offline-eval.md`
> (the implementation plan) before acting.

## Your task

Execute the remaining pipeline **using superpowers:subagent-driven-development** (fresh
implementer subagent per task → task review (spec + quality) → fix loop → broad final
review). The SDD ledger lives at
`.superpowers/sdd/2026-08-27-aucp-offline-eval/progress.md` — it is git-ignored, is the
authoritative record of what is already done, and is your recovery map. Read it first and
resume from the first task without a `complete` line. Do **not** re-run completed phases.

Remaining tasks, in order: **4.4 → 5.1 → 5.3 → 6.1 → Phase 7.** (Tasks 4.1–4.3, 4.5, and
5.2 are already complete and reviewed; the compiler and harness pure-logic modules exist
and are tested.)

## What is already done (do not redo)

- **Phase 0 gate:** endpoints + held-out set + cutoff approved by the human (below).
- **Phase 1:** eval contract frozen — `config/compile_config.json`, `config/heldout_exclusions.json`,
  `eval/tasks/{T1,T2,T3}.json` (each with a re-derived `task_definition_hash`).
- **Phase 2:** `notebooks/02_extract_atomic_memories.py` + `scripts/run_extract.sh` ran.
  `atomic_memories` holds **159 memories** from 22 eligible pre-cutoff sessions;
  `heldout_rows=0`.
- **Phase 3:** `notebooks/03_embed_memories.py` + `scripts/run_embed.sh` ran. All **159
  rows carry a 1024-dim `databricks-gte-large-en` embedding** + `embedding_model`;
  `remaining_null=0`; contamination re-assert passed.
- **Compiler pure-logic (Tasks 4.1–4.3):** `src/compiler/score.py` (`score`, `select`,
  `tie_break_key`), `arms.py` (`pool`), `render.py` (`render` → `RenderResult`), all
  unit-tested.
- **Harness pure-logic (Tasks 4.5, 5.2):** `src/harness/inject.py` (`stage_memory`,
  `verify_injection`), `metrics.py` (`exploratory_reads_before_edit`,
  `failed_test_cycles`), all unit-tested.
- **Test suite:** 137 tests, all green (`pytest -q` from repo root). Keep it green.

## Frozen decisions / exact values (authoritative — do not invent)

- **Catalog/schema:** `ai_fde_hackathon_catalog.automatic_user_context_profiles`.
- **Tables:** `atomic_memories` (populated + embedded), `memory_artifacts` (Task 4.4
  writes), `eval_runs` (Task 5.1 writes). Schemas: SPEC §3.
- **Endpoints:** embedding `databricks-gte-large-en` (1024-dim); extraction
  `databricks-claude-sonnet-4-5`; **fixed eval agent = Claude Code pinned to
  `databricks-claude-sonnet-4-5`.**
- **CUTOFF_TS:** `2026-08-17T00:00:00Z` (strict `<`). Held-out ids: the 8 in
  `compile_config.json` `heldout_conversation_ids` ∪ every `conversation_ids` list in
  `heldout_exclusions.json`.
- **Base commit for eval tasks:** `ac6a62b2c84fd228844ea0c1e0af726d2ac74bcc`
  (`starting_commit` in the task manifests), on the accessible platform codebase.
- **Arms (for the 12 artifacts):** `empty`, `static_generic`, `retrieved`, `placebo`
  → 3 tasks × 4 arms = **12** frozen `MEMORY.md` artifacts.
- **T1 acceptance is an ORACLE:** the reference test suite from solution commit
  `98b8bd7` (`tests/unit/test_image_selection_auto_k.py`, 45 tests) is supplied at eval
  time as a **read-only fixture kept OUT of the agent's context** — never agent-authored
  tests, never revealed to the eval agent. T1's `/goal` pins the public API so the oracle
  binds; it must not reveal the algorithm.

## Execution mechanics (Databricks — mirror the existing pattern exactly)

- Notebooks run as **serverless one-time runs** via a `scripts/run_*.sh` wrapper that:
  stages `src/*` (and configs) to the UC Volume
  `dbfs:/Volumes/ai_fde_hackathon_catalog/automatic_user_context_profiles/raw/...` (the
  notebook adds the staged parent to `sys.path`), imports the notebook, submits with
  `databricks jobs submit --no-wait --json` (CLI profile `hackathon`), waits, and
  validates the `dbutils.notebook.exit(<json>)` **sentinel** via
  `scripts/_check_sentinel.py`. Copy `scripts/run_extract.sh` / `scripts/run_embed.sh`
  as the template for any new runner.
- **Every write notebook gets a `dry_run` widget defaulting to `"true"` (safe: no
  writes).** Run the dry-run gate first, inspect the exit JSON, then flip `dry_run="false"`
  for the billable/shared-write full run. Only the literal string `"false"` enables writes.
- **Contamination guard is mandatory in every notebook that touches a shared table:** an
  explicit `if heldout_rows != 0: raise RuntimeError(...)` (never `assert` — stripped
  under `python -O`) joining the table against the banned-id set. Copy the block from
  `02`/`03`.
- Watch for the SQL-string footgun fixed in `03`: never put a `# noqa` (or any `#`
  comment) *inside* a `spark.sql(f"""...""")` triple-quoted string — Spark SQL treats a
  leading `#` as a syntax error.

## The remaining tasks

### Task 4.4 — Compile + freeze the 12 artifacts (`notebooks/04_compile_memory_md.py`)
The plan lists this as "Modify", but the notebook does not exist yet — **create it**
(it wraps the already-tested `src/compiler/*` logic). For each `(task ∈ {T1,T2,T3}) ×
(arm ∈ {empty,static_generic,retrieved,placebo})`:
1. Read the embedded rows from `atomic_memories` (apply `compile_config.json`
   `metadata_filter`: `username=abdullah.said`, `domain_in=[repo:platform,general_workflow]`,
   `datetime < cutoff_ts`).
2. Embed the task's `/goal` query text with `databricks-gte-large-en` for the `retrieved`
   and `placebo` arms (same endpoint; batch of 3 queries is fine).
3. Run `score`→`select`→`arms.pool`→`render` to produce the byte-deterministic
   `MEMORY.md`, write the `.md` to the UC Volume, and insert a `memory_artifacts` row
   (SPEC §3.4: `payload_sha256`, `file_sha256`, `sentinel`, `selected_memory_ids`,
   `token_count`, `volume_path`, `compiler_version`, `config_json`).
4. **Assert 12 artifacts written; assert a second compile reproduces identical
   `file_sha256` for every `(task,arm)`** (byte-for-byte reuse — the trust anchor).
5. Sentinel `compile:OK` with the 12 shas. Add `scripts/run_compile.sh`.
Commit `feat(04): freeze 12 artifacts + hashes to memory_artifacts`.

### Task 5.1 — Run driver (`notebooks/05_run_eval.py` + `src/harness/run.py`)
Per `task × arm × repeat`: checkout the platform repo at `starting_commit` into a
throwaway worktree; `stage_memory`; launch the fixed Claude-Code agent
(`databricks-claude-sonnet-4-5`) on `goal_prompt`; **require the agent to echo the
sentinel first → set `injection_verified`; assert the staged `file_sha256` matches the
artifact (abort the run on mismatch).** After the run compute
`success = acceptance green AND regression green AND forbidden_files unchanged`; capture
`final_commit`, `elapsed_seconds`, `total_tool_calls`, `exploratory_reads_before_edit`,
`failed_test_cycles`, tokens, `trace_path`, `final_diff_path`. Enforce budgets
(`max_minutes`, `max_tool_calls`) → `failure_reason` on breach. Append one `eval_runs`
row (SPEC §3.5); save transcript + diff to the Volume. Commit
`feat(05): controlled run driver + eval_runs logging`.

### Task 5.3 — Execute the matrix (SPEC §8 fallback)
Run **T1 first** (fully offline `pytest` oracle acceptance) across all 4 arms × ≥2
repeats — this is the credible paired comparison; de-risk here. Run T2/T3 only if
`autonomy_dev` + `/Volumes` data are available; else fall back to recorded traces (SPEC
§8). Prefer one credible paired task over uncontrolled anecdotes. Verify `eval_runs` row
count; every row `injection_verified=true` with matching sha.

### Task 6.1 — Report (`notebooks/06_report.py`)
Success rate per `(task,arm)`; among successes, median `exploratory_reads_before_edit`,
tool calls, failed-test cycles, wall seconds, tokens. Per-task paired deltas: `retrieved`
vs `empty`/`static_generic`/`placebo`. **Falsifiable check: `retrieved > placebo`; flag
if `retrieved ≤ placebo`.** Label every table/chart "directional (N=3, one correlated
feature week)". Commit `feat(06): eval report — paired deltas + medians`.

### Phase 7 — Final verification (CHECKLIST)
Re-run `00_ingest` → assert idempotency (0 net new rows). Verify every table schema vs
SPEC §3. Verify from stored evidence: `atomic_memories` has 0 held-out-cluster rows and
all `source_datetime < cutoff_ts`. Verify the 12 `file_sha256` values and byte-for-byte
repeat reuse in `eval_runs`. Then run `/review` (Isaac Review) on the PR and cross-review
with a different vendor; **merge is a human action.**

## Hard constraints (contamination invariants — never violate)

- Temporal cutoff applied (strict `<`); held-out ids + their broader clusters excluded
  everywhere; **never** extract or compile solution-equivalent info for a held-out task;
  the T1 oracle test stays out of the eval agent's context.
- Do not modify treatment files, the frozen configs, or the frozen task manifests.
- Default to safe: `dry_run` defaults true; no destructive deletes without an explicit
  config flag; run the dry-run gate before every billable/shared-write full run.
- Pause for the human at each remaining live-infrastructure boundary (Phase 5 needs
  `autonomy_dev` access and the platform repo checked out) — flag blockers, don't guess
  around missing access.
