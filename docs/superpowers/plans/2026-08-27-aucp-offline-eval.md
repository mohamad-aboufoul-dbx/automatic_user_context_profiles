# AUCP Offline-Frozen Retrieval Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract tool-agnostic memory from Abdullah's pre-cutoff coding-agent sessions, compile it into frozen per-`(task, arm)` `MEMORY.md` artifacts, and run one fixed coding agent on 3 held-out UES4 `/goal` tasks to measure whether retrieved memory lowers repository-rediscovery cost versus empty/generic/placebo arms.

**Architecture:** Databricks + Unity Catalog pipeline (notebooks `02`–`06`) reading the already-ingested `raw_conversations_abdullah_said`, feeding the shared `atomic_memories` contract, then a deterministic offline compiler that freezes hashed `MEMORY.md` files, then a controlled Claude-Code eval harness that runs each frozen artifact byte-for-byte and logs to `eval_runs`.

**Tech Stack:** Python 3.11+, PySpark on Databricks (CLI profile `hackathon`), Unity Catalog (`ai_fde_hackathon_catalog.automatic_user_context_profiles`), a Databricks serving embedding endpoint, an extraction LLM endpoint, Claude Code as the fixed eval agent, `pytest` for the pure-logic compiler/harness units.

**Spec:** `docs/SPEC.md` (source of truth) + `docs/heldout-approval-packet.md` (the held-out contract awaiting sign-off). The plan argues from both; executors read all three.

## Global Constraints

- **UC namespace (verbatim):** `ai_fde_hackathon_catalog.automatic_user_context_profiles`; CLI profile `hackathon`.
- **Shared contract is frozen:** the `atomic_memories` columns + `memory_type` enum (SPEC §3.2) are the cross-team contract — never change columns or the 8-value enum.
- **CUTOFF_TS = `2026-08-17T00:00:00Z`** (proposed; must be human-approved before Phase 2).
- **Base commit for all 3 held-out tasks = `ac6a62b2c84fd228844ea0c1e0af726d2ac74bcc`** (`~/Projects/platform`, branch `image-selection`).
- **Contamination invariants (HARD):** apply the temporal cutoff before extraction; exclude every held-out `conversation_id` and its full feature cluster (packet §4); never extract a held-out task's final solution or solution-equivalent info; do not modify treatment files after a run; no live retrieval during a scored run.
- **Determinism:** one `config/compile_config.json` compiles every arm; same file byte-for-byte across repeats; every artifact carries `file_sha256` + sentinel.
- **`smart-bites` is excluded entirely** from every stage.
- **Human-input values still TODO** (block the phases that use them): `embedding_model` serving-endpoint id; extraction LLM endpoint id; the fixed eval agent model id. Do not invent these — get them from Abdullah.
- **Reporting honesty:** N=3, one correlated feature week; report per-task paired deltas + medians, labeled directional.

---

## Phase 0 — Land Day-1 and open the gate (human-gated)

### Task 0.1: Merge PR #2 (Day-1 ingest + sessionize) into `dev/asaid`

**Files:**
- Brings in: `notebooks/00_ingest_abdullah_tab.py`, `notebooks/01_sessionize.py`, `notebooks/_setup_uc_objects.py`, `scripts/run_day1.sh`, `scripts/_check_sentinel.py`, `data/raw_abdullah_tab.csv` (currently on branch `dev/asaid-day1`).

- [ ] **Step 1:** Confirm review is complete (`pr2-review.diff` / `pr2-fix-review.diff` addressed; Day-1 commits `6e2a317…b8d66a2` already resolve the review blockers).
- [ ] **Step 2 (human action):** merge `dev/asaid-day1` → `dev/asaid`.

```bash
git -C ~/Projects/memory/automatic_user_context_profiles checkout dev/asaid
git -C ~/Projects/memory/automatic_user_context_profiles merge --no-ff dev/asaid-day1
```

- [ ] **Step 3:** Verify `notebooks/00_*`, `01_*`, `data/raw_abdullah_tab.csv` now exist on `dev/asaid`; CHECKLIST items through "live setup → ingest → sessionize" stay checked.

### Task 0.2: Obtain held-out gate approval

- [ ] **Step 1:** Present `docs/heldout-approval-packet.md` §"Requires explicit human confirmation" (4 items) to Abdullah.
- [ ] **Step 2 (human action):** record explicit approval of the cutoff, base commit, T1/T2/T3 set, exclusion IDs, and the standalone-prompt rewrites.
- [ ] **Step 3:** Do NOT proceed to Phase 2 until this is recorded. (CHECKLIST STOP gate.)

---

## Phase 1 — Freeze the eval contract (allowed once Phase 0.2 approved; no extraction yet)

### Task 1.1: Populate `config/compile_config.json`

**Files:**
- Modify: `config/compile_config.json`

**Interfaces:**
- Produces: the frozen compiler config consumed by every Phase-4 task.

- [ ] **Step 1:** Replace the TODO stubs with approved values (leave `embedding_model` as the real endpoint id once known):

```json
{
  "embedding_model": "<APPROVED-serving-endpoint-id>",
  "cutoff_ts": "2026-08-17T00:00:00Z",
  "heldout_conversation_ids": [
    "d8d5b84d-1ddb-4c04-952a-3d071d5223c6", "06147d1c-537a-4ac7-b210-43c5909ad7b3", "15616d81-e31e-432c-b395-859f97a3689f",
    "3b41f7ea-1a52-4b31-b27b-e1e9cb1d1a34",
    "c7b01e88-4076-4997-984a-5ec488d6b942", "d7537515-d25c-4ad5-9cc7-d9fd979245b7", "cdef3eb6-12fd-42f5-98fb-34139a6d3195",
    "486c522b-61b0-425e-8669-280abfa203f3"
  ],
  "metadata_filter": { "username": "abdullah.said", "domain_in": ["repo:platform", "general_workflow"], "datetime_lt": "cutoff_ts" },
  "score_weights": { "semantic": 0.75, "recency": 0.15, "confidence": 0.10 },
  "recency_halflife_days": 120, "top_k": 12, "token_budget": 1200,
  "tie_breaker": ["confidence_desc", "source_datetime_desc", "memory_id_asc"],
  "render_template_version": "v1"
}
```

- [ ] **Step 2:** Keep the full feature-cluster exclusion (packet §4) in a sibling `config/heldout_exclusions.json` (the `heldout_conversation_ids` above is the tight set; the cluster+temporal cutoff is the real guard). Include the shared UES4 cluster + pre-cutoff 2025 review flags.
- [ ] **Step 3:** Commit `config: freeze cutoff + held-out exclusion set (approved)`.

### Task 1.2: Write `eval/tasks/T1.json`, `T2.json`, `T3.json`

**Files:**
- Create: `eval/tasks/T1.json`, `eval/tasks/T2.json`, `eval/tasks/T3.json`

**Interfaces:**
- Produces: rows matching `eval_tasks` schema (SPEC §3.3), consumed by Phase 4 (compile) + Phase 5 (run).

- [ ] **Step 1:** For each task fill: `task_id`, `goal_prompt` (the standalone rewrite from packet §1), `repository` (`~/Projects/platform`), `starting_commit` = `ac6a62b2c84fd228844ea0c1e0af726d2ac74bcc`, `temporal_cutoff` = cutoff, `heldout_conversation_ids` (that task's cluster), `allowed_domains` = `["repo:platform","general_workflow"]`, `acceptance_command`, `regression_command`, `required_files`, `forbidden_files`, `max_minutes`, `max_tool_calls`, `task_definition_hash` = sha256 of the canonicalized JSON minus the hash field.
- [ ] **Step 2 (T1 acceptance, objective):** `acceptance_command` = `pytest tests/unit/test_image_selection_auto_k.py -q`; `forbidden_files` = everything outside `src/frame_generation/image_selection/` + `tests/unit/`.
- [ ] **Step 3 (T2/T3 acceptance, structural):** `acceptance_command` = a notebook-parse + section-execution check on `autonomy_dev` (workspace `5515503609081527`); mark `regression_command` null; note the live-workspace dependency.
- [ ] **Step 4:** Compute `task_definition_hash`; commit `eval: add held-out task manifests T1–T3 (frozen)`.

---

## Phase 2 — Extraction `02` (GATED: requires Phase 0.2 approval + cutoff applied FIRST)

### Task 2.1: Contamination pre-filter (temporal cutoff + exclusion) — TDD

**Files:**
- Create: `notebooks/02_extract_atomic_memories.py`
- Create: `tests/unit/test_extract_prefilter.py`

**Interfaces:**
- Consumes: sessionized view `sessions_abdullah_said` (from `01`), `config/compile_config.json`.
- Produces: `eligible_sessions(df, cfg) -> df` — sessions whose `MAX(event_datetime) < cutoff_ts` AND `conversation_id ∉ heldout_conversation_ids` AND not in the cluster-exclusion list.

- [ ] **Step 1: Write the failing test** (pure function over a small DataFrame-like list of dicts):

```python
def test_prefilter_drops_after_cutoff_and_heldout():
    sessions = [
        {"conversation_id": "keep-1", "ended_at": "2026-08-16T23:59:59Z"},
        {"conversation_id": "drop-cutoff", "ended_at": "2026-08-17T00:00:00Z"},
        {"conversation_id": "d8d5b84d-1ddb-4c04-952a-3d071d5223c6", "ended_at": "2026-06-01T00:00:00Z"},
    ]
    cfg = {"cutoff_ts": "2026-08-17T00:00:00Z",
           "heldout_conversation_ids": ["d8d5b84d-1ddb-4c04-952a-3d071d5223c6"]}
    kept = [s["conversation_id"] for s in eligible_sessions(sessions, cfg)]
    assert kept == ["keep-1"]
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/unit/test_extract_prefilter.py -v` → FAIL (`eligible_sessions` undefined).
- [ ] **Step 3: Implement `eligible_sessions`** (strict `<` cutoff; drop exact held-out ids AND the cluster list from `heldout_exclusions.json`).
- [ ] **Step 4: Run** → PASS. Add a boundary test: `ended_at == cutoff_ts` is dropped (strict `<`).
- [ ] **Step 5: Commit** `feat(02): contamination pre-filter with tests`.

### Task 2.2: Structured extraction call + JSON-schema validation — TDD on the validator

**Files:**
- Modify: `notebooks/02_extract_atomic_memories.py`
- Create: `tests/unit/test_memory_schema.py`
- Uses: `prompts/extraction_prompt.md` (already written, v1).

**Interfaces:**
- Produces: `validate_memory(obj) -> list[str]` (returns violation messages; empty = valid), and `extract_session(transcript, meta, cfg) -> list[dict]` (calls the extraction LLM; integration, not unit-tested).

- [ ] **Step 1: Write failing test** for the validator:

```python
def test_memory_schema_enforces_enum_and_length():
    bad = {"memory_text": "x", "memory_type": "not_an_enum", "domain": "repo:platform",
           "evidence": "step 4: ...", "confidence": 1.5}
    errs = validate_memory(bad)
    assert any("memory_type" in e for e in errs)
    assert any("memory_text" in e for e in errs)   # <15 chars
    assert any("confidence" in e for e in errs)     # >1
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement `validate_memory`**: enum ∈ the 8 SPEC types; `15 ≤ len(memory_text) ≤ 500`; `0 ≤ confidence ≤ 1`; `evidence` non-empty and references a chat_step; reject arrays >20.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5:** Wire `extract_session` to call the approved extraction endpoint with the v1 prompt, parse JSON, run `validate_memory`, drop invalid rows with a logged reason. Add a `dry_run` mode that runs the prompt over ONE eligible session and prints the JSON (no table write) — the leakage eyeball.
- [ ] **Step 6: Commit** `feat(02): extraction + schema validation (dry-run gate)`.

### Task 2.3: Deterministic `memory_id` + write to `atomic_memories` — TDD on the id

**Files:**
- Modify: `notebooks/02_extract_atomic_memories.py`
- Create: `tests/unit/test_memory_id.py`

**Interfaces:**
- Produces: `memory_id(conversation_id, normalized_text, memory_type) -> str` = `sha256(f"{conversation_id}|{normalized_text}|{memory_type}")` hexdigest.

- [ ] **Step 1: Write failing test**: same inputs → same id; different `memory_type` → different id; normalization (lowercase/strip/collapse-ws) makes `" Foo  Bar "` and `"foo bar"` collide.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `memory_id` with explicit normalization; assert stability.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5:** Write rows to `atomic_memories` with all SPEC §3.2 columns (`extraction_model`, `extraction_run_id`, `created_at`, `source_datetime`, `evidence`, `confidence`; `embedding`/`embedding_model` null here — filled in Phase 3). Idempotent MERGE on `memory_id`.
- [ ] **Step 6:** Run extraction over ALL eligible sessions; assert `atomic_memories` non-empty and that `SELECT count(*) WHERE conversation_id IN (heldout)` = 0 (leakage assertion).
- [ ] **Step 7: Commit** `feat(02): write atomic_memories + zero-heldout assertion`.

---

## Phase 3 — Embedding `03`

### Task 3.1: Embed `memory_text` and backfill

**Files:**
- Create: `notebooks/03_embed_memories.py`

**Interfaces:**
- Consumes: `atomic_memories` rows with null `embedding`; the approved embedding endpoint.
- Produces: `embedding ARRAY<FLOAT>` + `embedding_model` populated; idempotent (only embeds rows missing an embedding).

- [ ] **Step 1:** Batch-call the embedding endpoint over `memory_text`; write `embedding` + `embedding_model` back via MERGE on `memory_id`.
- [ ] **Step 2:** Optional cheap dedup: within a `memory_type`, drop near-dupes at cosine ≥ 0.95 keeping highest `confidence` (SPEC §4). Guard behind a config flag; log dropped ids.
- [ ] **Step 3:** Assert every non-dropped row has a non-null embedding of the expected dimension; assert re-running embeds 0 new rows (idempotency).
- [ ] **Step 4: Commit** `feat(03): embed atomic_memories (idempotent) + optional dedup`.

---

## Phase 4 — Deterministic compile + injection `04` (heavy TDD — this is the trust anchor)

### Task 4.1: Retrieval scoring — TDD

**Files:**
- Create: `notebooks/04_compile_memory_md.py`
- Create: `src/compiler/score.py`
- Create: `tests/unit/test_compiler_score.py`

**Interfaces:**
- Produces: `score(query_vec, mem, cfg, now) -> float` = `0.75*cosine + 0.15*0.5**(age_days/halflife) + 0.10*confidence`; `select(mems, query_vec, cfg, now) -> list[mem]` applying `top_k`, `token_budget`, and `tie_breaker`.

- [ ] **Step 1: Write failing test** for `score` with a hand-computed expected value and for tie-break ordering (`confidence_desc`, then `source_datetime_desc`, then `memory_id_asc`).

```python
def test_score_formula_matches_spec():
    mem = {"embedding": [1.0, 0.0], "confidence": 0.5, "source_datetime": "2026-08-01T00:00:00Z"}
    cfg = {"score_weights": {"semantic":0.75,"recency":0.15,"confidence":0.10}, "recency_halflife_days":120}
    now = "2026-08-17T00:00:00Z"   # 16 days old
    s = score([1.0, 0.0], mem, cfg, now)
    assert abs(s - (0.75*1.0 + 0.15*0.5**(16/120) + 0.10*0.5)) < 1e-9
```

- [ ] **Step 2: Run** → FAIL. **Step 3:** Implement `score`/`select`. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(04): retrieval scoring + selection with tests`.

### Task 4.2: Per-arm candidate pools — TDD

**Files:**
- Create: `src/compiler/arms.py`
- Create: `tests/unit/test_compiler_arms.py`

**Interfaces:**
- Consumes: `score`/`select`.
- Produces: `pool(arm, mems, task, cfg, now) -> list[mem]` for arm ∈ {`empty`, `static_generic`, `retrieved`, `placebo`}.

- [ ] **Step 1: Write failing tests:** `empty` → `[]`; `static_generic` → top by `confidence·recency`, one-per-domain cap, query-independent; `retrieved` → uses task `/goal` as query; `placebo` → only non-`repo:platform` domains, count/tokens matched to `retrieved`, and asserts `repo:platform` never appears.
- [ ] **Step 2: Run** → FAIL. **Step 3:** Implement. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(04): per-arm candidate pools with tests`.

### Task 4.3: `MEMORY.md` render (v1) + sentinel + hashes — TDD

**Files:**
- Create: `src/compiler/render.py`
- Create: `tests/unit/test_compiler_render.py`

**Interfaces:**
- Produces: `render(task_id, arm, mems, cfg) -> (markdown, file_sha256, payload_sha256, sentinel, token_count)`; `sentinel = f"mem-{task_id}-{arm}-{shortsha}"`; template = SPEC §5 v1; empty sections omitted; `empty` arm = header + sentinel only.

- [ ] **Step 1: Write failing test:** rendering the same `(task_id, arm, mems, cfg)` twice yields identical `file_sha256`; the sentinel line is present and matches the pattern; an empty-arm render contains the sentinel and no memory bullets; a section with no memories is omitted.

```python
def test_render_is_byte_deterministic():
    out1 = render("T1", "retrieved", MEMS, CFG)
    out2 = render("T1", "retrieved", MEMS, CFG)
    assert out1.file_sha256 == out2.file_sha256
    assert out1.sentinel.startswith("mem-T1-retrieved-")
```

- [ ] **Step 2: Run** → FAIL. **Step 3:** Implement render + `sha256`. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(04): deterministic MEMORY.md render + sentinel + hash`.

### Task 4.4: Freeze artifacts to UC volume + `memory_artifacts` row

**Files:**
- Modify: `notebooks/04_compile_memory_md.py`

- [ ] **Step 1:** For each `(task ∈ {T1,T2,T3}) × (arm ∈ {empty,static_generic,retrieved,placebo})`, compile once and write the `.md` to the UC volume; insert a `memory_artifacts` row (SPEC §3.4) with `payload_sha256`, `file_sha256`, `sentinel`, `selected_memory_ids`, `token_count`, `volume_path`, `compiler_version`, `config_json`.
- [ ] **Step 2:** Assert 12 artifacts written; assert re-running `04` reproduces identical `file_sha256` for every `(task,arm)` (byte-for-byte reuse).
- [ ] **Step 3: Commit** `feat(04): freeze 12 artifacts + hashes to memory_artifacts`.

### Task 4.5: Agnostic loader + injection/sentinel verify — TDD

**Files:**
- Create: `src/harness/inject.py`
- Create: `tests/unit/test_inject.py`

**Interfaces:**
- Produces: `stage_memory(md_path, workdir)` → writes canonical `MEMORY.md` + one-line `CLAUDE.md` (`@MEMORY.md`); `verify_injection(agent_echo, expected_sentinel, staged_path, expected_sha) -> bool` (sentinel echoed AND `file_sha256(staged_path)==expected_sha`).

- [ ] **Step 1: Write failing test:** `verify_injection` returns False on sentinel mismatch, False on sha mismatch, True when both match; every arm (incl. `empty`) uses the SAME loader — only file contents differ.
- [ ] **Step 2: Run** → FAIL. **Step 3:** Implement. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(04): agnostic loader + sentinel/sha injection verify`.

---

## Phase 5 — Eval harness + controlled runs `05`

### Task 5.1: Run driver (one fixed agent, per `task × arm × repeat`)

**Files:**
- Create: `notebooks/05_run_eval.py`
- Create: `src/harness/run.py`

**Interfaces:**
- Consumes: `eval/tasks/*.json`, `memory_artifacts`, `stage_memory`/`verify_injection`.
- Produces: one `eval_runs` row per run (SPEC §3.5).

- [ ] **Step 1:** For each run: checkout `~/Projects/platform` at `starting_commit` into a throwaway worktree; `stage_memory`; launch the fixed Claude-Code agent on `goal_prompt`; require the agent to echo the sentinel first → set `injection_verified`; assert `file_sha256` matches artifact (abort run on mismatch).
- [ ] **Step 2:** After the run: compute `success = acceptance_command green AND regression green AND forbidden_files unchanged`; capture `final_commit`, `elapsed_seconds`, `total_tool_calls`, `exploratory_reads_before_edit`, `failed_test_cycles`, tokens, `trace_path`, `final_diff_path`.
- [ ] **Step 3:** Enforce budgets (`max_minutes`, `max_tool_calls`); on breach set `failure_reason` and stop the run.
- [ ] **Step 4:** Append to `eval_runs`; save transcript + diff to the UC volume.
- [ ] **Step 5: Commit** `feat(05): controlled run driver + eval_runs logging`.

### Task 5.2: Metric extraction from traces — TDD

**Files:**
- Create: `src/harness/metrics.py`
- Create: `tests/unit/test_metrics.py`

**Interfaces:**
- Produces: `exploratory_reads_before_edit(trace) -> int` (count of read/search tool calls before the first edit to a `required_file`); `failed_test_cycles(trace) -> int`.

- [ ] **Step 1: Write failing test** over a small synthetic trace (list of tool-call dicts) with a hand-counted expected value for reads-before-first-edit.
- [ ] **Step 2: Run** → FAIL. **Step 3:** Implement. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(05): trace metric extraction with tests`.

### Task 5.3: Execute the matrix (with SPEC fallback)

- [ ] **Step 1:** Run **T1** (fully offline `pytest` acceptance) across all 4 arms × ≥2 repeats first — this is the credible paired comparison; de-risk everything here.
- [ ] **Step 2:** Run **T2/T3** if `autonomy_dev` + `/Volumes` data are available; else fall back to recorded traces (SPEC §8 fallback). Prefer one credible paired task over uncontrolled anecdotes.
- [ ] **Step 3:** Verify `eval_runs` has the expected row count; every row has `injection_verified=true` and matching sha.

---

## Phase 6 — Report `06`

### Task 6.1: Per-task paired deltas + medians

**Files:**
- Create: `notebooks/06_report.py`

- [ ] **Step 1:** From `eval_runs`: success rate per `(task,arm)`; among successful runs, median `exploratory_reads_before_edit`, tool calls, failed-test cycles, wall seconds, tokens.
- [ ] **Step 2:** Per-task paired deltas: `retrieved` vs `empty`, vs `static_generic`, vs `placebo`. The falsifiable check: `retrieved > placebo`; flag if `retrieved ≤ placebo`.
- [ ] **Step 3:** Render tables/charts labeled "directional (N=3, one correlated feature week)".
- [ ] **Step 4: Commit** `feat(06): eval report — paired deltas + medians`.

---

## Phase 7 — Final verification (CHECKLIST "Final verification")

- [ ] **Step 1:** Re-run `00_ingest` → assert idempotency (0 net new rows).
- [ ] **Step 2:** Verify every table schema against SPEC §3.
- [ ] **Step 3:** Verify contamination guards from stored evidence: `atomic_memories` has 0 held-out-cluster rows; all `source_datetime < cutoff_ts`.
- [ ] **Step 4:** Verify the 12 `file_sha256` values and byte-for-byte repeat reuse in `eval_runs`.
- [ ] **Step 5:** Run `/review` (Isaac Review) on the implementation PRs; cross-review with a different vendor; merge on approval (human action).

---

## Self-Review (spec coverage)

- SPEC §3 tables → Task 0.1 (`_setup_uc_objects`), 2.3, 3.1, 4.4, 5.1. ✅
- §4 extraction (cutoff, exclusion, prompt, schema, dedup) → Tasks 2.1–2.3, 3.1. ✅
- §5 deterministic compile (scoring, arms, render, sentinel, hash, freeze) → Tasks 4.1–4.4. ✅
- §6 injection + harness + metrics → Tasks 4.5, 5.1–5.3. ✅ (agnostic loader; same loader every arm.)
- §6 report → Task 6.1. ✅
- §8 fallback → Task 5.3. ✅
- §9 risks (injection silent-fail, artifact freeze, N=3 over-claim, leakage) → Tasks 4.3/4.5 (sentinel+sha), 4.4 (freeze), 6.1 (directional label), 2.1/2.3 (cutoff+zero-heldout). ✅

**Open gaps requiring human input (not code gaps):** `embedding_model` endpoint id, extraction LLM endpoint id, fixed eval-agent model id, `autonomy_dev` access for T2/T3. All blocked-pending, flagged in Global Constraints.
