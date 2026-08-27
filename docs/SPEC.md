# Build Spec — Automatic User Context Profiles: Agnostic Memory + Offline-Frozen Retrieval Eval

**Owner:** Abdullah · **Duration:** 2-day hackathon · **Repo:** `automatic_user_context_profiles` (branch: `dev/asaid`) · **UC namespace:** `ai_fde_hackathon_catalog.automatic_user_context_profiles`

> Direction: everyone works independently and syncs on the final day. The memory artifact is **tool-agnostic** by design (Abdullah's work is typically done outside a Genie space). Genie appears only as an optional transportability epilogue, never as an evaluator.

---

## 1. Objective + falsifiable claim

**Objective:** Show that memory distilled from Abdullah's past coding-agent sessions, injected as a **tool-agnostic `MEMORY.md`**, lets a fresh coding agent solve a held-out `/goal` task with less repository-rediscovery and fewer steps — without sacrificing correctness.

**The one falsifiable claim:**
> On held-out `/goal` coding tasks in Abdullah's repo, a fixed coding agent given the **retrieved task-specific `MEMORY.md`** reaches tests-green at an equal-or-higher success rate AND with lower repository-rediscovery cost than the **empty** arm, **and beats the matched same-user placebo** (equal token budget, irrelevant memories). If retrieved ≤ placebo, the claim is falsified.

**Non-goals:** knowledge graph · live retrieval during a scored run · multi-user modeling · production serving · automatic memory mutation · Genie as evaluator · the 5,480-row cross-agent corpus (optional extension) · statistical significance from 3 tasks.

---

## 2. Architecture + how it fits the team

```
Abdullah tab (Chatbot conversations)
      ↓ direct load
UC raw_conversations_abdullah_said
      ↓ sessionize
LLM structured extraction
      ↓
UC atomic_memories (+ embeddings)   ← shared foundation / team contract
      ├── Generic user profile      → teammate's Genie profile output → eval arm: static_generic
      └── Task-specific retrieval    → your differentiating treatment  → eval arm: retrieved
      ↓ deterministic compiler
frozen MEMORY.md + SHA-256 (per task × arm)
      ↓
one fixed coding agent, controlled runs
      ↓
tests + traces → UC eval_runs
```

**Reconciliation with the team:** the teammate's "automatic user context profile" **is** the `static_generic` arm (and a producer into the shared `atomic_memories` table). Your retrieved, task-specific `MEMORY.md` is the treatment being tested against it. **Genie is a memory producer, never the evaluator** for coding tasks. Each team member uploads **only their own** chat history; convergence happens at the final-day sync via the shared `atomic_memories` schema.

---

## 3. Unity Catalog data model

> The raw table is **user-scoped** so uploads don't collide in the shared schema (each teammate uploads their own). `atomic_memories` matches this schema exactly — it is the shared contract.

```sql
-- 3.1 Raw conversations — loaded straight from the Abdullah tab (no spreadsheet-first step)
CREATE TABLE IF NOT EXISTS ai_fde_hackathon_catalog.automatic_user_context_profiles.raw_conversations_abdullah_said (
  username STRING NOT NULL, query_text STRING, response_text STRING,
  chat_step BIGINT NOT NULL, conversation_id STRING NOT NULL, source_tool STRING NOT NULL,
  event_datetime TIMESTAMP NOT NULL, ingested_at TIMESTAMP NOT NULL,
  source_name STRING NOT NULL, row_hash STRING NOT NULL   -- SHA-256 of normalized fields → idempotent MERGE
) USING DELTA TBLPROPERTIES ('delta.enableChangeDataFeed'='true');
-- column map: Username→username, Query→query_text, Response→response_text, Chat_step→chat_step,
-- Conversation_id→conversation_id, Tool→source_tool, Datetime→event_datetime;
-- source_name='chatbot_conversations_abdullah'; filter to username='abdullah.said'

-- 3.2 Atomic memories (SHARED CONTRACT — match columns + enum exactly across the team)
CREATE TABLE IF NOT EXISTS ai_fde_hackathon_catalog.automatic_user_context_profiles.atomic_memories (
  memory_id STRING NOT NULL,   -- SHA-256(conversation_id | normalized_text | memory_type)  (deterministic)
  username STRING NOT NULL, memory_text STRING NOT NULL, memory_type STRING NOT NULL,
  domain STRING NOT NULL, source_tool STRING NOT NULL, conversation_id STRING NOT NULL,
  source_datetime TIMESTAMP NOT NULL, evidence STRING NOT NULL, confidence DOUBLE NOT NULL,
  embedding ARRAY<FLOAT>, embedding_model STRING, extraction_model STRING NOT NULL,
  extraction_run_id STRING NOT NULL, created_at TIMESTAMP NOT NULL
) USING DELTA TBLPROPERTIES ('delta.enableChangeDataFeed'='true');
-- memory_type ∈ {user_preference, workflow_preference, repository_fact, architecture_decision,
--                command_or_environment, failure_and_fix, code_convention, project_state}

-- 3.3 Eval tasks (pre-declared; frozen)
CREATE TABLE IF NOT EXISTS ai_fde_hackathon_catalog.automatic_user_context_profiles.eval_tasks (
  task_id STRING NOT NULL, goal_prompt STRING NOT NULL, repository STRING NOT NULL,
  starting_commit STRING NOT NULL, temporal_cutoff TIMESTAMP NOT NULL,
  heldout_conversation_ids ARRAY<STRING> NOT NULL, allowed_domains ARRAY<STRING> NOT NULL,
  acceptance_command STRING NOT NULL, regression_command STRING,
  required_files ARRAY<STRING>, forbidden_files ARRAY<STRING>,
  max_minutes INT NOT NULL, max_tool_calls INT NOT NULL,
  task_definition_hash STRING NOT NULL, created_at TIMESTAMP NOT NULL
) USING DELTA;

-- 3.4 Compiled/frozen artifacts (one row per task × arm)
CREATE TABLE IF NOT EXISTS ai_fde_hackathon_catalog.automatic_user_context_profiles.memory_artifacts (
  artifact_id STRING NOT NULL, task_id STRING NOT NULL, arm STRING NOT NULL,
  compiler_version STRING NOT NULL, config_json STRING NOT NULL,
  selected_memory_ids ARRAY<STRING> NOT NULL, payload_sha256 STRING NOT NULL,
  file_sha256 STRING NOT NULL, sentinel STRING NOT NULL, memory_markdown STRING NOT NULL,
  token_count INT NOT NULL, volume_path STRING NOT NULL, frozen_at TIMESTAMP NOT NULL
) USING DELTA;   -- arm ∈ {empty, static_generic, retrieved, placebo, oracle}

-- 3.5 Eval runs (append-only)
CREATE TABLE IF NOT EXISTS ai_fde_hackathon_catalog.automatic_user_context_profiles.eval_runs (
  run_id STRING NOT NULL, task_id STRING NOT NULL, arm STRING NOT NULL, repeat_number INT NOT NULL,
  artifact_id STRING NOT NULL, file_sha256 STRING NOT NULL,
  sentinel_expected STRING NOT NULL, sentinel_observed STRING, injection_verified BOOLEAN NOT NULL,
  agent_name STRING NOT NULL, model_name STRING NOT NULL,
  starting_commit STRING NOT NULL, final_commit STRING,
  success BOOLEAN NOT NULL, acceptance_tests_passed BOOLEAN NOT NULL,
  regression_tests_passed BOOLEAN, forbidden_files_unchanged BOOLEAN NOT NULL,
  elapsed_seconds DOUBLE, total_tool_calls INT,
  exploratory_reads_before_edit INT, failed_test_cycles INT,
  input_tokens BIGINT, output_tokens BIGINT,
  trace_path STRING, final_diff_path STRING, failure_reason STRING,
  started_at TIMESTAMP NOT NULL, completed_at TIMESTAMP
) USING DELTA;
```

---

## 4. Extraction (prompt + guards)

**Contamination guards — apply BEFORE extraction:**
1. Temporal cutoff `CUTOFF_TS` — only conversations whose `MAX(event_datetime) < CUTOFF_TS` are eligible.
2. Explicit exclusion of `HELDOUT_CONVERSATION_IDS`.
3. Never extract final solutions to a held-out task.

Sessionize by `conversation_id` ordered by `event_datetime, chat_step`; preserve `/goal` text, commands, paths, branches, worktrees, tests, failures, corrections; don't merge separate conversation_ids.

**Structured output:** array (≤20) of `{memory_text (15–500 chars), memory_type (enum), domain, evidence (with chat_step), confidence 0–1}`.

**Extraction prompt:**
```
SYSTEM
You extract durable, reusable MEMORIES about one user (Abdullah) and their projects
from a single coding-agent session, to give a fresh agent that has NEVER seen this
user or repo. Extract only what generalizes beyond this one task. Output ONLY JSON
matching the schema.

KEEP: user_preference · workflow_preference (TDD-first, worktrees, branch naming,
commit/PR style) · command_or_environment (exact test/build/lint/run commands, setup)
· repository_fact (module layout, key files) · architecture_decision (with rationale)
· failure_and_fix (a mistake + its correction — high value) · code_convention ·
project_state (label temporary/in-flight state as such).

DROP: generic programming advice · one-off trivia · whole-session summaries ·
secrets/tokens/credentials/PII · large code snippets · claims found ONLY in an
assistant response unless the user accepted it OR code/tests/tool-output confirmed it.

RULES: atomic & self-contained (no pronouns without referent); tool-neutral language;
preserve concrete paths/commands/test names/ticket IDs/branches; don't infer a stable
preference from one weak example; evidence must quote/paraphrase the transcript with
its chat_step; return {"memories":[]} if nothing durable. Do not summarize.

USER
username={{username}} conversation_id={{conversation_id}} tool={{source_tool}}
started={{started_at}} ended={{ended_at}}
Transcript:
{{transcript}}
```

Embed each `memory_text` with the frozen embedding model; write to `atomic_memories`. Optional cheap dedup: drop near-duplicates within a `memory_type` at cosine ≥ 0.95, keep highest confidence.

---

## 5. Deterministic compile (freeze + hash)

**Frozen config — commit as `config/compile_config.json`, declare ONCE:**
```json
{ "embedding_model": "<fixed-endpoint-id>", "cutoff_ts": "<ISO8601>",
  "heldout_conversation_ids": ["...","...","..."],
  "metadata_filter": {"username":"abdullah.said","domain_in":["repo:<name>","general_workflow"],"datetime_lt":"cutoff_ts"},
  "score_weights": {"semantic":0.75,"recency":0.15,"confidence":0.10},
  "recency_halflife_days": 120, "top_k": 12, "token_budget": 1200,
  "tie_breaker": ["confidence_desc","source_datetime_desc","memory_id_asc"],
  "render_template_version": "v1" }
```
Score = `0.75·cosine(query,mem) + 0.15·0.5^(age_days/120) + 0.10·confidence`. **No hand-editing after selection. Same config compiles every arm.**

**Per-arm candidate pools (same template, same token budget):**
- **empty** — header + sentinel only, zero memories.
- **static_generic** — ignore the query; top memories by `confidence·recency`, one-per-domain cap (this is the teammate's profile).
- **retrieved** — query = the task `/goal` text; run the scoring above.
- **placebo** — same user, same token budget, memories from *unrelated* domains (e.g. `ffmpeg`, `sfm`), explicitly excluding the target `repo:` domain; matched count/tokens to retrieved.
- **oracle (optional)** — hand-picked ideal memories for ONE task (upper bound; only if Day-2 slack).

**`MEMORY.md` render template (`v1`):**
```markdown
<!-- MEMORY_SENTINEL: {{sentinel}} -->
# MEMORY.md
## How to use this file
Prior context about the user (Abdullah) and their projects, distilled from past sessions.
Treat as trusted background. Prefer its commands, conventions, and file locations over
re-exploring. Do not restate it. Before acting, acknowledge memory version {{sentinel}}.

## User & Working Style        {{preference + workflow_preference}}
## Commands & Environment       {{command_or_environment}}
## Project / Repo Facts         {{repository_fact}}
## Decisions & Rationale        {{architecture_decision}}
## Known Pitfalls & Fixes       {{failure_and_fix}}
## Conventions                  {{code_convention}}
## In-Flight / Superseded State {{project_state}}
```
Empty sections omitted. `sentinel = mem-{task_id}-{arm}-{shortsha}`. After render: compute `file_sha256`, write the `memory_artifacts` row, **reuse that file byte-for-byte for every repeat of that (task, arm).**

---

## 6. Injection + eval harness

- **Injection (agnostic + native):** canonical `MEMORY.md`; Claude Code loads it via a one-line `CLAUDE.md` containing `@MEMORY.md`. **Every arm uses the same loader; only file contents vary** (empty arm gets a header-only file). Verify per run: agent echoes the sentinel → set `injection_verified`; assert `file_sha256` matches the artifact.
- **Task:** one held-out `/goal` against a pinned `starting_commit`. `success = acceptance_command green AND regression green AND no forbidden_files touched`. Budget: `max_minutes` / `max_tool_calls`.
- **Metrics — success FIRST, efficiency only on successful runs:** primary efficiency = `exploratory_reads_before_edit` (repository-rediscovery cost); plus total tool calls, failed test cycles, wall seconds, tokens.
- **Design:** ~3 tasks × 4 arms (empty / static_generic / retrieved / placebo), ≥2 repeats each; log everything to `eval_runs`; transcripts + diffs to a UC volume. Report per-task paired deltas + medians; label as directional given N.

---

## 7. Repo layout (branch `dev/asaid`, slots beside `dev/maboufoul`)

```
automatic_user_context_profiles/
├── docs/SPEC.md                          # this file
├── notebooks/
│   ├── 00_ingest_abdullah_tab.py         # sheet/CSV → raw_conversations_abdullah_said (idempotent MERGE)
│   ├── 01_sessionize.py
│   ├── 02_extract_atomic_memories.py     # shared producer (feeds teammate's profile too)
│   ├── 03_embed_memories.py
│   ├── 04_compile_memory_md.py           # deterministic; writes memory_artifacts + volume files
│   ├── 05_run_eval.py                    # drives the coding agent per (task,arm,repeat)
│   └── 06_report.py                      # tables + charts from eval_runs
├── config/compile_config.json
├── prompts/extraction_prompt.md
├── eval/tasks/T1.json, T2.json, T3.json
└── README.md
```
Keep `02_extract_*` and the `atomic_memories` schema as the **shared contract** with the teammate's `dev/maboufoul` profile notebooks.

---

## 8. Day plan (with slip fallback)

**Day 1 — get one clean comparison by EOD.**
- H1: create schema objects + `00_ingest` (Abdullah tab → `raw_conversations_abdullah_said`, filtered to `abdullah.said`).
- H2: `01_sessionize`; pick `CUTOFF_TS` + 3 held-out `/goal` tasks + pin commits.
- H3–5: `02_extract` + `03_embed` → populate `atomic_memories`; eyeball for leakage.
- H5–6: `04_compile` for the empty + retrieved arms of ONE task; wire the `@MEMORY.md` loader + sentinel check.
- H6–8: run that one task, empty vs retrieved, end-to-end (de-risks everything).

**Day 2 — replicate + arms + demo.**
- H1–3: compile + run remaining tasks × all 4 arms (add `static_generic` + `placebo`), ≥2 repeats.
- H3–5: `06_report`; screen-record the cleanest retrieved-vs-empty task.
- H5–7: build the 3-min demo (memory card → side-by-side → scorecard → placebo mic-drop).
- H7–8: buffer + optional Genie transportability epilogue (≤15s, no metrics).

**Fallback if you slip:** ONE task, arms empty + retrieved + placebo only (drop static_generic and oracle first); if live agent re-runs are flaky, fall back to recorded traces. *One credible paired task beats ten uncontrolled anecdotes.*

---

## 9. Risks + alignment talking points

**Risks:** injection silently fails (mitigate: sentinel + hash assert) · retrieval quality confounds the result (mitigate: freeze artifacts, add oracle arm if time) · N=3 over-claim (mitigate: report as directional) · extraction leaks held-out solutions (mitigate: temporal cutoff + id exclusion) · agent nondeterminism (mitigate: repeats + pinned model/commit).

**Alignment talking points for the final-day sync (no turf fight):**
1. **One funnel, not two.** The Genie profile-builder and the retrieval path both read/write the *same* `atomic_memories` table — profile is a *producer*, compile is a *consumer*.
2. **The profile is literally the `static_generic` arm.** The eval needs a generic-profile baseline; the teammate's Genie profile fills that slot — it becomes a scored arm, not a side quest.
3. **Genie stays a producer, not the judge.** For coding tasks there's no tests-green metric inside Genie, so evaluation runs in one fixed coding agent; Genie appears only as an optional transportability epilogue.
4. **One repo, per-user branches, one contract.** Agree on the `atomic_memories` columns + `memory_type` enum; then each branch moves independently.
5. **The demo tells both stories:** history → shared memory table → (their) generic profile *and* (my) task-specific retrieval → measurable coding speedup + portability.

---

## Open decisions
- Confirm the `atomic_memories` schema with the team (shared contract) at/ before the sync.
- Confirm the user-scoped raw table naming convention (`raw_conversations_abdullah_said` is the safe default).
- Whether to spend Day-2 slack on the oracle arm.
