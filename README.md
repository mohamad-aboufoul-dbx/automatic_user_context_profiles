# automatic_user_context_profiles — `dev/asaid`

Automatic User Context Profiles: **tool-agnostic agent memory** + an
**offline-frozen retrieval evaluation**. Full build spec: [`docs/SPEC.md`](docs/SPEC.md).

## What this proves

On held-out `/goal` coding tasks in Abdullah's repo, a fixed coding agent given the
**retrieved task-specific `MEMORY.md`** reaches tests-green at an equal-or-higher
success rate AND with lower repository-rediscovery cost than the **empty** arm, **and
beats the matched same-user placebo**. If retrieved ≤ placebo, the claim is falsified.

## Hard invariants

- Tool-agnostic `MEMORY.md`; Claude Code loads it only via a one-line `CLAUDE.md` → `@MEMORY.md`.
- No live retrieval inside a scored run — retrieval is an **offline compile** that
  **freezes + hashes** a `MEMORY.md` per (task, arm), reused byte-for-byte across repeats.
- Every arm uses the **same loader**; only file **contents** vary (empty arm = header-only).
- Contamination guards before extraction: temporal cutoff + exclude held-out
  `conversation_ids`; never extract final solutions to a held-out task.
- Success measured **before** efficiency; never hand-edit a treatment file after a run.

## Layout (§7)

```
├── docs/SPEC.md                          # the build spec (source of truth)
├── notebooks/
│   ├── 00_ingest_abdullah_tab.py         # Abdullah tab/CSV → raw_conversations_abdullah_said (idempotent MERGE)
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

## Data scoping

- Shared schema: `ai_fde_hackathon_catalog.automatic_user_context_profiles`
- Raw table is **user-scoped**: `raw_conversations_abdullah_said` — contains ONLY
  Abdullah's history (teammates upload their own).
- `atomic_memories` matches the §3.2 schema exactly — it is the **shared team contract**.

## Status

Day-1 in progress. Extraction (`02`) is gated on sign-off of the temporal cutoff +
held-out `/goal` task set (protects the contamination guard).
