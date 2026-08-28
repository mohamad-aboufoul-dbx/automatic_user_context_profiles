# Task 6.1 Report — Eval Report: Paired Deltas + Medians

**Status:** DONE  
**Branch:** `asaid/aucp-phase4-7`

---

## What was built

### `src/report/__init__.py`
Package init.

### `src/report/aggregate.py`
Pure, side-effect-free aggregation module (no SQL, no network). Public API:

- `compute_report(rows: list[dict]) -> ReportData` — top-level entry point
- `render_report(report: ReportData) -> str` — renders markdown string
- `compute_arm_summaries(rows) -> dict[(task_id, arm), ArmSummary]` — Step 1
- `compute_paired_deltas(summaries, task_ids) -> list[PairedDelta]` — Step 2
- `compute_falsifiable_checks(summaries, task_ids) -> list[FalsifiableCheckResult]` — Step 3

### `notebooks/06_report.py`
Thin local entrypoint. Reads `eval_runs` via the SQL statement-execution API
(`warehouse_id=41659c95dacd3bf0`, profile `hackathon`), calls
`compute_report(rows)` and `render_report(report)`. Outputs to stdout and
optionally to `--output PATH`. Optional matplotlib charts via `--charts-dir DIR`
(lazy import; tests never need matplotlib).

### `tests/unit/test_report_aggregate.py`
54 unit tests covering all specified scenarios.

---

## Computation details

### Step 1 — Per (task, arm) summary
- `success_rate = n_success / n_runs`; `n_runs = 0` → 0.0
- Medians computed over **successful runs only**
- `med_tokens` = median of `(input_tokens + output_tokens)` per successful run
- No successful runs → all median fields `None`

### Step 2 — Paired deltas
- For each task: `retrieved` minus each of `{empty, static_generic, placebo}`
- `success_rate_delta` = retrieved_rate − comparator_rate (positive = retrieved better)
- Efficiency deltas: retrieved_median − comparator_median (negative = retrieved leaner)
- Either arm absent → `None` deltas

### Step 3 — Falsifiable check (headline)
Verdict per task: **PASS** if `retrieved > placebo`; **FLAG** if `retrieved ≤ placebo`.

**Tie-breaking rule (explicit and documented in module docstring):**  
On a success-rate tie, compare efficiency metrics in priority order:
1. `exploratory_reads_before_edit`
2. `total_tool_calls`
3. `failed_test_cycles`

First metric with a strict difference decides:
- `retrieved < placebo` → **PASS**, basis = `efficiency-tiebreak (<metric>)`
- `retrieved > placebo` → **FLAG**, basis = `efficiency-tiebreak (<metric>)`
- All equal or unavailable → **FLAG**, basis = `efficiency-tiebreak (all-tied)`

Missing arm data → **FLAG**, basis = `missing-data`.

The `basis` column in the rendered table records which signal decided each task
so the reader always knows whether success-rate or efficiency settled the verdict.

### Step 4 — Rendering
Every section and table carries the label "directional (N=3, one correlated feature week)".
Empty `eval_runs` → "No runs yet" placeholder, no crash.

---

## Test coverage (54 tests, all passing)

| Scenario | Test(s) |
|---|---|
| Success rate 0/N, K/N, N/N | `test_*_success_*` |
| Median among successes only | `test_medians_only_from_successes` |
| No successes → all medians None | `test_no_successful_runs_yields_all_none_medians` |
| Combined token median | `test_combined_token_median`, `test_token_median_with_none_fields` |
| Paired deltas math + missing arm | `TestComputePairedDeltas` |
| (a) retrieved > placebo → PASS | `test_retrieved_beats_placebo_on_rate` |
| (b) retrieved < placebo → FLAG | `test_retrieved_below_placebo_on_rate` |
| (c) tie → efficiency wins PASS | `test_rate_tie_broken_by_exploratory_reads_retrieved_wins` |
| (d) tie → efficiency wins FLAG | `test_rate_tie_broken_by_exploratory_reads_retrieved_loses` |
| (e) first equal, second decides | `test_rate_tie_first_metric_equal_second_decides` |
| (e2) first two equal, third decides | `test_rate_tie_first_two_equal_third_decides` |
| (f) all tied → FLAG all-tied | `test_all_tiebreak_metrics_equal_yields_flag` |
| (g) missing arm → FLAG missing-data | `test_missing_retrieved/placebo_arm_*` |
| (d) empty input → no crash | `test_empty_input_returns_empty_report` |
| Full 3×4 matrix → PASS per task | `test_full_matrix_three_tasks_four_arms` |
| Render: no-runs message | `test_empty_report_renders_no_runs_message` |
| Render: three sections present | `test_non_empty_report_has_three_sections` |
| Render: directional / N=3 labels | `test_directional_label_*`, `test_n_equals_label_*` |
| Render: no "None" in output | `test_em_dash_for_none_medians` |
| Render: tiebreak documented | `test_tiebreak_basis_is_documented_in_table` |

---

## Concerns / known limitations

1. **No live data yet.** The matrix (Task 5.3) has not run; `eval_runs` will be empty or T1-only when this report is first invoked. The notebook handles this gracefully with the "no runs yet" message.
2. **N=3 is inherently directional.** All tables are labeled accordingly. No statistical significance claims are made anywhere.
3. **`string_true_success_is_coerced` test confirms** that SQL string booleans are handled correctly; the `_coerce_row` function in the notebook handles the same.
4. **matplotlib is optional.** The `--charts-dir` flag saves a success-rate PNG; if matplotlib is absent the notebook still runs cleanly (lazy import).
