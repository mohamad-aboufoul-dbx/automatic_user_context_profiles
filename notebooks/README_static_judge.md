# Static judge-only eval (`07_eval_static_judge.py` + `08_report_static_judge.py`)

Scores an eval set of **pre-captured** chatbot answers with an LLM judge and asks a
single question: **did injecting a user-context profile improve answer quality?**

"Static" means nothing is re-generated. The answers were captured earlier under
five conditions and are read from a table as-is; the notebooks only judge and
report on them. No live agent is called.

## Judge design: holistic coverage rubric (not element counting)

The judge scores each answer with a single **holistic coverage rubric**, returning
`coverage_score` on a fixed 5-point scale:

| score | meaning |
| --- | --- |
| 1.0 | fully covers essentially all expected content, correct specifics |
| 0.75 | covers most; minor gaps |
| 0.5 | covers about half; notable gaps |
| 0.25 | touches the topic but misses most |
| 0.0 | off-topic / wrong / does not address it |

It also returns a `failure_mode` (`covers_well` / `partial` / `too_generic` /
`off_topic` / `wrong_specifics`) and a short rationale.

**Why holistic, not element-counting:** the ground truth
(`Expected_elements_in_response`) is free text — comma-separated phrases whose
commas are internal punctuation, so there is no reliable delimiter to split on.
Asking the judge to count "how many of N expected elements are covered" makes the
denominator N non-deterministic (the same expected text parses into a different
element count run to run), which makes the score unstable. Grading coverage
holistically against a fixed rubric removes that dependence and produces stable,
comparable scores.

## `07_eval_static_judge.py` — score

1. Reads the eval-set table (one row = one answer under one condition).
2. Decodes `Profile_used` into `data_combo` (`B` / `C`) and `profile_level`
   (`level_2_detailed` / `level_3_evidence`). `N/a` means *no profile* and is the
   per-question baseline.
3. Per row, calls a Foundation Model chat endpoint with the holistic rubric above
   and parses the JSON result defensively (strips ```` ``` ```` code fences and
   extracts the first balanced JSON object; a Databricks Claude endpoint quirk).
   A row that cannot be scored records `parse_error` and keeps `coverage_score`
   null rather than failing the run.
4. Writes per-row results to `OUTPUT_TABLE` (a Delta table).

`q_key = len(Expected_elements_in_response)` is a coarse per-question fingerprint
used to align the same question across conditions (the query text itself changes
when a profile is injected, so it cannot be the key).

## `08_report_static_judge.py` — report

Reads the results table and prints: (a) mean coverage by `Profile_used`;
(b) by `data_combo` × `profile_level`; (c) by question age × condition;
(d) **delta vs the `N/a` baseline** per condition; (e) failure-mode counts.

## Widgets

| Widget | Default | Notes |
| --- | --- | --- |
| `DRY_RUN` | `true` | `true` = no endpoint calls, no table write; runs on a tiny synthetic sample so the notebook is importable/compilable with no workspace |
| `INPUT_TABLE` | the eval-set table | read-only (07) |
| `OUTPUT_TABLE` / `RESULTS_TABLE` | `..._eval_results_static_judge` | overwritten each run (07 writes, 08 reads) |
| `JUDGE_ENDPOINT` | `databricks-claude-sonnet-4` | FM chat endpoint used as the judge (07) |
| `ROW_LIMIT` | `0` | `0` = all rows (07) |

## Running

1. Open `07_eval_static_judge.py`, set `DRY_RUN=false`, confirm `INPUT_TABLE` /
   `JUDGE_ENDPOINT`, and run. It writes the per-row results table.
2. Open `08_report_static_judge.py`, set `DRY_RUN=false`, point `RESULTS_TABLE` at
   the same table, and run to see the by-condition summary and baseline deltas.

## Caveat

The current eval set is tiny (~3 questions × 5 conditions = 15 rows, zero
replicates). Treat any observed difference as directional only, not
statistically reliable — add replicates before drawing conclusions.
