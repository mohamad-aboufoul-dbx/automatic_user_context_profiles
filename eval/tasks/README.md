# Eval task fixtures

These fixtures provide labeled ground truth for the judge-loop Q&A evaluation.
Each question is run in two arms: one with retrieved user context and one without
it. A judge compares the answer against `expected_answer_elements` and can use
`grounding` to audit every required element.

## Fixture schema

Each `T<number>.json` file contains one object with:

- `id`: stable task identifier matching the filename.
- `question`: a new question that benefits from prior context.
- `expected_answer_elements`: specific required facts or decisions.
- `grounding`: committed repository sources supporting those elements.
- `memory_type`: one value from the enum in `docs/SPEC.md` §3.2.
- `task_area`: a short grouping label for reporting.
- `rationale`: why the with-context arm should need fewer follow-ups.
- `safety`: fixture-level confirmation that sensitive content was excluded.

`grounding.conversation_ids` refers to records in
`data/raw_abdullah_tab.csv`. `grounding.repo_files` lists any additional
committed documents used. The `support` list paraphrases the evidence used to
derive the expected answer; it is not additional ground truth.

## Assumptions

The build spec defines the Unity Catalog `eval_tasks` execution table but does
not define a JSON schema for the newer judge-loop Q&A fixtures. These files use
a minimal auditable schema centered on the fields required by the approved
judge-loop design. They intentionally omit coding-run fields such as
`starting_commit`, `acceptance_command`, and `max_tool_calls`, because these are
answer-quality tasks rather than repository mutation runs.

Grounding is limited to committed repository content. No Unity Catalog data or
external per-person corpus was available or used.
