# Automatic User Context Profiles

Users have a lot of context spread throughout their chats with agents, documentation they've set up, Slack threads, emails, etc. Whenever they start a new session with any chatbot, they practically have to start from scratch — explaining themselves and what they need. Add on to it that they often have to ask several follow-ups and/or argue with the agent to give them what they're looking for, and we find that a good chunk of the day can be wasted.

Automatic User Context Profiles consolidate information throughout a user's chat history, documentation, and more to give agents the right context before moving forward. The goal is to give agents a better understanding of where the user's at, their overall goals, and the question behind the question, so they can respond more effectively.

---

## Repository Structure

```
automatic_user_context_profiles/
├── profile_builder          # Databricks notebook: iterative profile generation pipeline
├── profiles/                # All profile outputs, prefix, docs, and build script
│   ├── prefix.yaml          # Agent instruction prefix (prepended to every profile)
│   ├── build_profiles.py    # Script to rebuild all 12 files from UC
│   ├── USAGE_GUIDE.md       # Detailed architecture + runtime usage documentation
│   ├── {user_slug}_combo_a_level_0_cold_start.yaml
│   ├── {user_slug}_combo_a_level_1_compact.yaml
│   ├── {user_slug}_combo_a_level_2_detailed.yaml
│   ├── {user_slug}_combo_a_level_3_evidence.yaml
│   ├── {user_slug}_combo_b_level_0_cold_start.yaml
│   ├── ...                           # (12 files per user)
│   └── {user_slug}_combo_c_level_3_evidence.yaml
└── README.md                # This file
```

---

## Quick Start: Using Profiles with Agents

Each file in `profiles/` is self-contained — it includes both the **agent instruction prefix** (how to interpret the profile) and the **profile YAML** (the actual user context). Pick the right file for your use case:

| File | When to use |
|------|-------------|
| `{user}_combo_c_level_1_compact.yaml` | **Default choice.** Paste into Glean, Perplexity, or any system prompt. Small (~8K chars), covers communication prefs, active projects, capabilities. |
| `{user}_combo_c_level_2_detailed.yaml` | For project-specific deep work. Includes interaction outcomes, project threads, customer context (~21K chars). |
| `{user}_combo_c_level_0_cold_start.yaml` | Minimal identity-only context. Use when you want the agent to know WHO you are but not infer preferences. |
| `{user}_combo_c_level_3_evidence.yaml` | For auditing/debugging profiles. Not meant for agent injection. Contains raw evidence records with provenance. |

Filenames are derived dynamically from the `user_name` column (e.g., "Mohamad Aboufoul" → `mohamad_aboufoul`). When multiple users exist in the checkpoint table, all profiles are built side-by-side.

**Combo A** = career ladder only (cold-start baseline)  
**Combo B** = career ladder + chat history (iterative)  
**Combo C** = career ladder + chat + project documents (fullest, recommended)  

### To paste into an agent tool:

1. Open the relevant `combo_*` YAML file
2. Copy the full contents
3. Paste as system prompt, custom instructions, or context prefix
4. The prefix tells the agent how to interpret capability levels, confidence scores, and interaction patterns

### To refresh after new data:

1. Run the `profile_builder` notebook (generates new profiles from updated chat/docs)
2. Run `profiles/build_profiles.py` from a notebook cell to rebuild all 12 files

---

## Architecture

Profiles are built **incrementally and chronologically** — one conversation or document chunk at a time:

```
Career Ladder → Cold-Start L0–L3            [Combo A]
    ↓
For each conversation (by date): update L1–L3    [Combo B]
    ↓
For each document chunk (by date): update L1–L3  [Combo C]
```

This design:
- Preserves temporal signal ("asked about X once" ≠ "expert in X")
- Tracks interaction outcomes (what worked, what failed, repair strategies)
- Avoids batch-inference errors where all context is dumped at once
- Supports checkpointing so session loss never discards progress

---

## Profile Levels

| Level | Name | Size | Purpose | Agent Injection? |
|-------|------|------|---------|------------------|
| 0 | Cold-Start | ~5–9K | Identity + role baseline from career ladder | Only before L1+ exists |
| 1 | Compact | ~4–8K | Communication prefs, active threads, capabilities | **Always** (system prompt) |
| 2 | Detailed | ~17–21K | Project threads, interaction outcomes, growth model | **On-demand** (by topic) |
| 3 | Evidence | ~22–26K | Atomic observations with provenance and confidence | **Never** (maintenance only) |

---

## Interaction Patterns

Following the Ontology PDF specification, profiles track:

- **interaction_outcomes**: What the agent said → what the user did → did it work?
- **failure_modes**: `buried_answer`, `too_verbose`, `too_basic`, `wrong_assumption`, `missing_example`, `premature_solutioning`
- **response_pattern_claims**: Distilled, reusable guidance (e.g., "for debugging, lead with the answer")
- **repair_strategies**: What to do differently next time

A single failure is a scoped observation, not a global preference change. Only repeated patterns or explicit feedback update the compact profile.

---

## Data Sources

| Source | Location | Role |
|--------|----------|------|
| Career ladder | `/Volumes/.../cold_start_context_documents/*.xlsx` | L5 role expectations (cold-start baseline) |
| Chat history | `chatbot_conversations_mohamad` UC table | 129 messages, 15 conversations (Jun–Aug 2026) |
| Project documents | `/Volumes/.../chat_additional_context_documents/*.pdf` | 4 PDFs (~670K chars total) |

## UC Tables

| Table | Purpose |
|-------|---------|
| `profile_checkpoints_v2` | Latest profile state per combo with step tracking |
| `user_context_profiles` | Published final profiles (copied from checkpoints) |
| `chatbot_conversations_mohamad` | Source chat data |

---

## Model

`databricks-claude-opus-4-8` (pay-per-token, strongest model available via `ai_query` for nuanced inference about capabilities vs. curiosity).

---

See `profiles/USAGE_GUIDE.md` for detailed architecture documentation, source precedence rules, and runtime integration patterns.