# Extraction Prompt (v1)

Distilled from `docs/SPEC.md` §4. This is the tool-agnostic memory extraction
prompt applied per session in `notebooks/02_extract_atomic_memories.py`.

Structured output: array (≤20) of
`{memory_text (15–500 chars), memory_type (enum), domain, evidence (with chat_step), confidence 0–1}`.

`memory_type ∈ {user_preference, workflow_preference, repository_fact, architecture_decision, command_or_environment, failure_and_fix, code_convention, project_state}`

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
