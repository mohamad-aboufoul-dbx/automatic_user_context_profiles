# Profile-Context Eval Harness

Measures whether injecting a user-context profile helps an agent answer eval
questions in **fewer follow-up turns** than running with no context at all.

Each eval task is run once per arm. After every agent turn an LLM judge checks
whether the answer covers all of the task's expected answer elements. If it
does not, the judge's list of missing elements becomes the next follow-up turn.
The number of turns needed (`iterations_to_answer`) is the primary metric.

## Arms

| Arm | Meaning |
| --- | --- |
| `no_context` | Control — the agent gets the question and no profile context |
| `level_0_cold_start` | Profile injected at level 0 |
| `level_1_compact` | Profile injected at level 1 |
| `level_2_detailed` | Profile injected at level 2 |
| `level_3_evidence` | Profile injected at level 3 |

Profile text is loaded at runtime from
`{CATALOG}.{SCHEMA}.user_context_profiles` (falling back to
`profile_checkpoints_v2`), keyed on `(user_name, profile_level, data_combo)`.
No profile bodies are stored in this repo.

## Data combos

| Combo | Inputs used to build the profile |
| --- | --- |
| `A` | cold-start only |
| `B` | cold-start + chat |
| `C` | cold-start + chat + docs |

## How to run

1. Attach `notebooks/05_run_eval.py` to a cluster.
2. Set the widgets:
   - `MAX_ITERATIONS` — max agent turns per task (default `5`)
   - `DATA_COMBO` — `A` / `B` / `C` (default `C`)
   - `AGENT_ENDPOINT`, `JUDGE_ENDPOINT` — serving endpoints
   - `CATALOG`, `SCHEMA`, `USER_NAME`
   - `DRY_RUN` — leave at `true` for the first run
3. Run with `DRY_RUN=true` first. Results are printed and displayed but nothing
   is written. Set `DRY_RUN=false` to append rows to
   `{CATALOG}.{SCHEMA}.eval_runs`.
4. Run `notebooks/06_report.py` to read `eval_runs` and get per-arm pass rate
   and mean iterations-to-answer, plus the `no_context` control compared
   against each level arm.

## Eval tasks

`05_run_eval` globs `eval/tasks/T*.json`. Each file is one task:

```json
{
  "id": "T000_example",
  "question": "...",
  "expected_answer_elements": ["...", "..."]
}
```

If no task files are found, a small inline example set is used so the harness
still runs end to end.

## Output schema (`eval_runs`)

| Column | Type | Notes |
| --- | --- | --- |
| `task_id` | string | |
| `arm` | string | one of the five arms |
| `data_combo` | string | `A` / `B` / `C` |
| `answered` | boolean | judge accepted the answer |
| `iterations_to_answer` | int | `NULL` when never answered within `MAX_ITERATIONS` |
| `final_answer_len` | int | length only — answer text is not persisted |
| `max_iterations` | int | the cap used for this run |
