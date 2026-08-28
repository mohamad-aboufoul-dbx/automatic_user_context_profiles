# Extraction Prompt (v1)

Distilled from `docs/SPEC.md` §4. This is the tool-agnostic memory extraction
prompt applied per session in `notebooks/02_extract_atomic_memories.py`.

Structured output: a JSON object `{"memories": [ ... ]}` with ≤20 items; each item
has EXACTLY the keys `{memory_text (15–500 chars), memory_type (enum), domain,
evidence (with chat_step), confidence 0–1}`.

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

OUTPUT SCHEMA: Return ONLY a JSON object of the form {"memories": [ ... ]} with at most
20 items (return {"memories":[]} if nothing durable). Each item MUST be an object with
EXACTLY these five keys — use these exact key names, do NOT use type/key/value/tags:
  - "memory_text": string, 15–500 characters, atomic and self-contained.
  - "memory_type": exactly one of user_preference, workflow_preference, repository_fact,
    architecture_decision, command_or_environment, failure_and_fix, code_convention,
    project_state.
  - "domain": a short string — use "repo:<name>" when the memory is about a specific
    repository (e.g. "repo:platform"), otherwise "general_workflow".
  - "evidence": a short quote/paraphrase from the transcript that INCLUDES the chat_step
    (e.g. "chat_step 2: USER asked ...").
  - "confidence": a number between 0.0 and 1.0.
Every item MUST include all five keys.

EXAMPLE item:
{"memory_text": "Abdullah works on class-agnostic object detection for IR/thermal imagery and wants bounding boxes for high-objectness regions without classification.", "memory_type": "project_state", "domain": "general_workflow", "evidence": "chat_step 2: USER asked about pretrained models like OWLv2 for IR data, class-agnostic detections", "confidence": 0.9}

USER
username={{username}} conversation_id={{conversation_id}} tool={{source_tool}}
started={{started_at}} ended={{ended_at}}
Transcript:
{{transcript}}
```
