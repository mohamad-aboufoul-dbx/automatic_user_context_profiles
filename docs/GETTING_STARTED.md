# Getting Started — Automatic User Context Profiles

Welcome! This guide takes a brand-new contributor from zero to a working Day-1
run of the **Automatic User Context Profiles** project. Read it alongside
[`SPEC.md`](SPEC.md) (the source of truth) and [`CHECKLIST.md`](CHECKLIST.md)
(build progress).

## 1. What this project is

Automatic User Context Profiles is two things bolted together:

1. **Tool-agnostic, per-user agent memory.** We distill a user's past
   coding-agent sessions into a single `MEMORY.md` that any coding agent can
   consume. The memory is *tool-agnostic*: it is plain Markdown, and a Claude
   Code agent loads it with a one-line `CLAUDE.md` containing `@MEMORY.md`. No
   agent-specific API, no live retrieval at runtime.
2. **An offline-frozen retrieval evaluation.** To prove the memory actually
   helps, we run a controlled experiment: for each held-out task we *compile
   and freeze* a `MEMORY.md` per experimental arm (offline, hashed, reused
   byte-for-byte), then drive one fixed coding agent against each frozen
   artifact and compare outcomes.

The user whose history we build from is **Abdullah** (`username = abdullah.said`).
Each teammate uploads only their own history; the team converges at a final sync
via a shared `atomic_memories` schema (see §4).

## 2. The one falsifiable claim

> On held-out `/goal` coding tasks in Abdullah's repo, a fixed coding agent
> given the **retrieved task-specific `MEMORY.md`** reaches tests-green at an
> **equal-or-higher success rate AND with lower repository-rediscovery cost**
> than the **empty** arm, **and beats the matched same-user placebo** (equal
> token budget, irrelevant memories). If retrieved ≤ placebo, the claim is
> falsified.

Two things to notice:

- **Success before efficiency.** Success (tests green, no forbidden files
  touched) is measured first; the efficiency win (fewer exploratory reads
  before the first edit) only counts on successful runs.
- **The placebo is the bar, not just the empty arm.** Beating an empty file is
  necessary but not sufficient — the retrieved memory must also beat a
  same-user, same-budget file full of *irrelevant* memories. This controls for
  "any context helps."

If retrieved does not beat the placebo, we say so plainly: the claim is
falsified.

## 3. Hard invariants

These are non-negotiable and recur throughout the pipeline:

- **Tool-agnostic `MEMORY.md`** loaded only via `CLAUDE.md` → `@MEMORY.md`.
- **No live retrieval inside a scored run.** Retrieval is an *offline compile*
  that freezes + hashes a `MEMORY.md` per (task, arm); the same bytes are
  reused for every repeat.
- **One loader for all arms** — only file *contents* vary. The empty arm gets a
  header-only file.
- **Contamination guards before extraction:** a temporal cutoff, explicit
  exclusion of held-out `conversation_id`s, and never extracting a held-out
  task's final solution.
- **Success measured before efficiency; never hand-edit a treatment file after a
  run.**

## 4. Unity Catalog schema and core tables

Everything lives in one Unity Catalog namespace:

```
catalog:  ai_fde_hackathon_catalog
schema:   automatic_user_context_profiles
```

`notebooks/_setup_uc_objects.py` creates all five tables idempotently
(`CREATE TABLE IF NOT EXISTS`) and verifies each live table against the SPEC §3
schema — it fails loudly on any column, type, or nullability drift. The five
core tables:

| Table | Purpose | Key idea |
|---|---|---|
| `raw_conversations_abdullah_said` | Raw chatbot conversations, **user-scoped** (Abdullah only) | Loaded from the frozen CSV via an idempotent `row_hash` (SHA-256) MERGE |
| `atomic_memories` | Distilled memories — the **shared team contract** | One row per memory; the schema + `memory_type` enum are fixed across the team |
| `eval_tasks` | Pre-declared, frozen held-out tasks | `goal_prompt`, pinned `starting_commit`, `temporal_cutoff`, held-out `conversation_id`s, acceptance/regression commands |
| `memory_artifacts` | One frozen `MEMORY.md` per (task × arm) | Stores `memory_markdown`, `file_sha256`, `sentinel`, `volume_path`; `arm ∈ {empty, static_generic, retrieved, placebo, oracle}` |
| `eval_runs` | Append-only run results | Success + efficiency metrics per `(task, arm, repeat)` |

A few details worth knowing on day one:

- **`raw_conversations_abdullah_said`** columns: `username, query_text,
  response_text, chat_step, conversation_id, source_tool, event_datetime,
  ingested_at, source_name, row_hash`. It is a Delta table with change data
  feed enabled. The `row_hash` (SHA-256 of normalized fields) drives an
  idempotent MERGE, so re-running ingest is safe.
- **`atomic_memories`** is the shared contract. `memory_type` is an enum:
  `user_preference`, `workflow_preference`, `repository_fact`,
  `architecture_decision`, `command_or_environment`, `failure_and_fix`,
  `code_convention`, `project_state`.
- A managed UC **volume** (`raw`) under the schema stages the frozen CSV at
  `dbfs:/Volumes/ai_fde_hackathon_catalog/automatic_user_context_profiles/raw/raw_abdullah_tab.csv`
  for the ingest notebook.

## 5. The notebook pipeline

The end-to-end flow, in order:

| # | Notebook | What it does | Present today? |
|---|---|---|---|
| setup | `_setup_uc_objects.py` | Create + schema-verify all 5 UC tables | ✅ Yes |
| 00 | `00_ingest_abdullah_tab.py` | Frozen CSV → `raw_conversations_abdullah_said` (idempotent MERGE) | ✅ Yes |
| 01 | `01_sessionize.py` | Group rows by `conversation_id`, order deterministically → one session per conversation | ✅ Yes |
| 02 | `02_extract_atomic_memories.py` | LLM extraction of atomic memories → `atomic_memories` (shared producer) | ❌ **Not yet in the tree** |
| 03 | `03_embed_memories.py` | Embed each memory with the frozen model | ❌ **Not yet in the tree** |
| 04 | `04_compile_memory_md.py` | Deterministically compile + freeze + hash a `MEMORY.md` per (task, arm) → `memory_artifacts` + volume files | ❌ **Not yet in the tree** |
| 05 | `05_run_eval.py` | Drive the fixed coding agent per (task, arm, repeat) | ❌ **Not yet in the tree** |
| 06 | `06_report.py` | Build tables + charts from `eval_runs` | ❌ **Not yet in the tree** |

> **Accuracy note — read this before you start coding.** Notebooks **02–06 are
> referenced in `README.md` and `docs/SPEC.md`** as the planned pipeline, but
> they are **not yet present in the repository**. `notebooks/` currently contains
> only `_setup_uc_objects.py`, `00_ingest_abdullah_tab.py`, `01_sessionize.py`,
> and a `.gitkeep`. Likewise, the task manifests `eval/tasks/T1.json, T2.json,
> T3.json` referenced in the spec are **not yet present** (`eval/tasks/` holds
> only a `.gitkeep`), and `config/compile_config.json` is still filled with
> `TODO` placeholders for the cutoff, held-out IDs, embedding model, and domains.
> This is intentional: extraction is gated on human sign-off (see §7).

What the present notebooks do:

- **`_setup_uc_objects.py`** — creates the schema + all five tables and verifies
  them against SPEC §3. Exits with sentinel `setup:OK`.
- **`00_ingest_abdullah_tab.py`** — default source is the **frozen committed
  CSV** staged in the UC volume (`INGEST_SOURCE=frozen_csv`); pulling live from
  Google Sheets is opt-in (`INGEST_SOURCE=sheets`) and not reproducible. It
  hard-asserts: row count in `[240, 260]` (~247 expected), all rows
  `username='abdullah.said'`, no unparseable datetimes, no empty
  `conversation_id`s, unique `row_hash`es pre- and post-MERGE. Exits with
  `ingest:OK`.
- **`01_sessionize.py`** — pure sessionization: groups by `conversation_id`,
  orders events by `(event_datetime ASC, chat_step ASC, row_hash ASC)`, emits
  exactly one session per conversation (verified), and publishes the temp view
  `sessions_abdullah_said` for downstream notebooks. No extraction here. Exits
  with `sessionize:OK`.

## 6. The eval arms

Each held-out task is run under several arms. Every arm uses the **same
loader** and the **same render template**; only the *contents* of the
`MEMORY.md` differ, and every artifact is frozen + hashed before any scored
run.

- **`empty`** — header + sentinel only, zero memories. The lower-bound
  baseline.
- **`static_generic`** — ignore the task query; take top memories by
  `confidence × recency` with a one-per-domain cap. This is the teammate's
  generic user profile, repurposed as a scored baseline.
- **`retrieved`** — the treatment. Query = the task's `/goal` text; memories
  are ranked by the scoring formula and selected under the token budget.
- **`placebo`** — same user, same token budget, but memories from **unrelated
  domains** (e.g. `ffmpeg`, `sfm`), explicitly excluding the target `repo:`
  domain; matched to `retrieved` on count/tokens. This is the bar the treatment
  must beat.
- **`oracle`** *(optional)* — hand-picked ideal memories for one task; an
  upper bound, only if there is Day-2 slack.

**Scoring** (from `config/compile_config.json`):

```
score = 0.75·cosine(query, memory) + 0.15·0.5^(age_days / 120) + 0.10·confidence
```

with `top_k = 12`, `token_budget = 1200`, tie-breaker `confidence_desc →
source_datetime_desc → memory_id_asc`, and render template `v1`. The same
config compiles every arm; nothing is hand-edited after selection.

## 7. How to run Day 1

"Day 1" means: stand up the UC objects, ingest Abdullah's frozen data, and
sessionize it. It deliberately **stops before extraction**, which is gated on
human approval.

### Prerequisites

- The **Databricks CLI** (`databricks`) and **Python 3** on your PATH.
- A Databricks CLI profile with access to the `ai_fde_hackathon_catalog`
  catalog. The runner defaults to profile `fe-ai-sage` (override it with the
  first argument). Authenticate once, for example:

  ```bash
  databricks auth login --host https://fe-ai-sage.cloud.databricks.com --profile fe-ai-sage
  ```

  The runner verifies auth with `databricks current-user me --profile <profile>`.

### Run it

From the repo root:

```bash
bash scripts/run_day1.sh            # uses the default profile
bash scripts/run_day1.sh my-profile # or pass a different profile
```

`scripts/run_day1.sh` does, in order:

1. **Verify auth** and derive the per-user workspace path
   `/Users/<you>/hackathon_auto_profiles`.
2. **Ensure the UC schema + managed `raw` volume exist**, then upload the
   frozen CSV (`data/raw_abdullah_tab.csv`) to the volume. (Public DBFS root is
   disabled on this workspace, so the volume is the staging area.)
3. **Import** `_setup_uc_objects`, `00_ingest_abdullah_tab`, and
   `01_sessionize` into the workspace.
4. **Run those three notebooks in order** — each as its own serverless one-time
   job, waiting for `SUCCESS`, and validating the exit sentinel (`setup:OK`,
   `ingest:OK`, `sessionize:OK`) via `scripts/_check_sentinel.py`. A run that
   exits without the expected sentinel is treated as a failure (it catches the
   "only the import cell ran" case).

The runner is safe to re-run: `CREATE TABLE IF NOT EXISTS` and the `row_hash`
MERGE are both idempotent.

### The STOP gate

The runner ends with a prominent **HARD STOP**. Extraction (notebook 02) is
**not** run by `run_day1.sh` and must not begin until a human has explicitly
signed off on **all** of:

1. `CUTOFF_TS` — the temporal contamination cutoff.
2. Three held-out `/goal` tasks.
3. Pinned starting commits + the `conversation_id` exclusions for each task.

Until that sign-off is recorded, do not run extraction, embedding,
compilation, or any scored run. This is a review checkpoint, not a green
light. (See `docs/CHECKLIST.md` → "Held-out evaluation gate — approval
required".)

### Sanity-check the sentinel validator locally (no Databricks needed)

The sentinel check has its own offline test:

```bash
bash scripts/test_validate_sentinel.sh
```

It pipes representative `jobs get-run-output` JSON into
`scripts/_check_sentinel.py` and asserts the pass/fail cases the runner
depends on.

## 8. Where things stand / what's next

Today the repo contains the **foundation only**: the spec, the checklist, the
setup/ingest/sessionize notebooks, the frozen CSV, the extraction prompt, the
compile config (with `TODO`s), and the Day-1 runner + sentinel tooling. The
extract → embed → compile → eval → report notebooks (02–06), the eval task
manifests, and the finalized compile config all land **after** the
contamination gate is approved.

If you are joining mid-flight:

1. Read `docs/SPEC.md` (the contract) and `docs/CHECKLIST.md` (what's done).
2. Run Day 1 (§7) to confirm the foundation is healthy in your workspace.
3. Check the checklist for the held-out evaluation gate — that is the next
   unblock.

---

*Source of truth: `docs/SPEC.md`. Update `docs/CHECKLIST.md` as work lands; the
spec and its hard invariants take precedence.*
