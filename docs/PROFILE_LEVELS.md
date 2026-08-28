# User-Context Profile Levels

A user-context profile is structured context an agent can request to better tailor its response. The four levels support progressive disclosure: lower levels are smaller and cheaper to carry, while higher levels provide richer context at a larger character cost.

Use two choices together:

1. **Profile level** controls the depth and structure of the returned profile.
2. **Data combo** controls which source categories contribute context.

Choose the lowest level and least-rich data combo that can answer the question, then escalate only when the task requires more context.

## Level 0: Cold Start

**Name:** `level_0_cold_start`

**Purpose:** Request this for baseline role and identity context when chat or document context is unnecessary. It is the stable starting point for context-aware behavior.

**YAML key structure:**

```text
cold_start_profile:
  identity:
  role_context:
```

**Size behavior:** Level 0 is always **4,636 characters** for data combos A, B, and C. Cold start ignores chat and documents, so richer combos do not increase its size.

## Level 1: Compact

**Name:** `level_1_compact`

**Purpose:** Request this when the agent needs a compact summary of the user and current focus without the size of detailed or evidence-oriented context.

**YAML key structure:**

```text
level_1_compact:
  compact_profile:
    about_user:
    current_focus:
```

**Size behavior:** Level 1 remains nearly flat across the data-combo axis: **4,041 characters** for A and **4,038 characters** for both B and C.

## Level 2: Detailed

**Name:** `level_2_detailed`

**Purpose:** Request this when a compact profile is insufficient and the task needs a more detailed user description. Prefer it for questions where additional context can materially change the answer.

**YAML key structure:**

```text
level_2_detailed:
  detailed_profile:
    about_me:
```

**Size behavior:** Level 2 grows substantially when contextual sources are added: **4,236 characters** for A, **16,105 characters** for B, and **17,069 characters** for C.

## Level 3: Evidence

**Name:** `level_3_evidence`

**Purpose:** Request this when the agent needs the richest profile representation and the task justifies the highest character cost. Use it selectively rather than as a default.

**YAML key structure:**

```text
updated_profiles:
  level_1:
  level_2:
```

The top-level structure contains nested level 1 and level 2 blocks.

**Size behavior:** Level 3 is the largest option in every combo: **7,773 characters** for A, **22,248 characters** for B, and **21,514 characters** for C.

## Data-Combo Axis

The data-combo axis is the input-richness knob: A is without conversational or document context, while B and C progressively add contextual source categories.

| Combo | Meaning | `data_sources` value |
|---|---|---|
| A | Cold-start only | `career_ladder` |
| B | Cold-start + Chat | `career_ladder+chat_history_iterative` |
| C | Cold-start + Chat + Docs | `career_ladder+chat_history+documents_iterative` |

The combo does not affect Level 0 because cold start ignores chat and documents. Level 1 also stays nearly constant. Levels 2 and 3 expand substantially with B and C, so richer input sources have their largest size impact at the detailed and evidence levels.

## Size Reference

Character counts are `profile_length_chars`.

| Profile level | Combo A | Combo B | Combo C |
|---|---:|---:|---:|
| `level_0_cold_start` | 4,636 | 4,636 | 4,636 |
| `level_1_compact` | 4,041 | 4,038 | 4,038 |
| `level_2_detailed` | 4,236 | 16,105 | 17,069 |
| `level_3_evidence` | 7,773 | 22,248 | 21,514 |

## Request Guidance

- Start with Level 0 for baseline role and identity context.
- Use Level 1 when a compact user summary and current focus are enough.
- Move to Level 2 when additional detail can change the answer.
- Reserve Level 3 for tasks that need the richest available profile representation.
- Start with combo A when chat and document context are not required.
- Add combo B or C only when those contextual sources are relevant to the task.
- Treat character count as the explicit cost signal: richer context is often larger, especially at Levels 2 and 3.
