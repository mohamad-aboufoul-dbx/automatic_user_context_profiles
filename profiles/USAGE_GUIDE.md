# Automatic User Context Profiles — Usage Guide

## What This Is

A system that builds a layered, chronologically-aware representation of a user from their career metadata, conversation history, and authored documents. The output is a set of YAML profiles designed to be injected into AI agent contexts (system prompts, retrieval layers) to reduce clarification turns and improve response relevance.

## Architecture Overview

```
Career Ladder → Cold-Start Profiles (L0–L3)  [Combo A]
    ↓
For each conversation (chronological): update L1–L3  [Combo B]
    ↓
For each document chunk (chronological): update L1–L3  [Combo C]
```

Each piece of evidence is processed one-at-a-time. The model sees the current profile state plus one new conversation or document chunk, and decides what (if anything) to update. This preserves temporal signal — "asked about X once" ≠ "expert in X."

---

## The 4 Profile Levels

| Level | Name | Size | Changes? | Purpose |
|-------|------|------|----------|---------|
| 0 | Cold-Start | ~4K chars | Never | Identity + role baseline from career ladder |
| 1 | Compact | ~4–8K chars | After each evidence | Quick personalization for every turn |
| 2 | Detailed | ~5–10K chars | After each evidence | Full project threads, interaction patterns, growth |
| 3 | Evidence | ~10–20K chars | After each evidence | Atomic observations with provenance and confidence |

---

## How to Use Profiles at Runtime

### Tier 1: Always-On Context (Level 1)

**Inject Level 1 directly into the system prompt on every turn.**

Level 1 is small (~4–8K chars) and contains:
- Communication preferences by task type
- Active projects and threads
- Capability snapshot (what to explain vs. what to skip)
- Response pattern claims (what works/fails for this user)

This gives any agent immediate personalization at negligible token cost.

### Tier 2: On-Demand Retrieval (Level 2)

**Retrieve relevant sections of Level 2 based on the conversation topic.**

When the user mentions a known project, customer, or task type:
1. Match the request against `project_threads` and `customer_context`
2. Pull the matching section from Level 2
3. Include it alongside Level 1 in the context

Level 2 also contains `interaction_outcomes` and `response_pattern_claims` — use these to guide response structure for specific task types.

### Tier 3: Profile Maintenance Only (Level 3)

**Level 3 is never injected into user-facing conversations.**

It's used by the profile builder itself to:
- Track evidence provenance (which conversation said what)
- Resolve conflicts between sources
- Apply time decay to stale claims
- Audit why a profile says what it says

### Level 0: Superseded

Once Level 1+ exists, Level 0 is historical only. It shows what the system knew before any behavioral evidence existed.

---

## Providing Profiles to External Tools (Glean, Perplexity, etc.)

When injecting a profile as context to an external AI tool:

1. Use the `profile_context_prefix.yaml` file as a preamble
2. Follow it with the Level 1 compact profile
3. Optionally append relevant Level 2 sections

Example system prompt construction:

```
[profile_context_prefix.yaml content]

---
[Level 1 compact profile YAML]
---

[Your actual task prompt]
```

The prefix tells the agent HOW to interpret the profile. The profile provides the WHO.

---

## Key Design Principles

### Conservative Inference
- One question about a topic → `still_learning` at most
- Multiple applied uses → `proficient_in`
- Repeated successful delivery + explicit feedback → `strongest_areas`

### Interaction Patterns (from Ontology PDF)
- Track what worked AND what failed
- Record failure modes: `buried_answer`, `too_verbose`, `too_basic`, `wrong_assumption`, etc.
- Observable signals only — no psychological conclusions
- A single failure is scoped, not global
- Repeated patterns distill into `response_pattern_claims`

### Chronological Awareness
- Earlier evidence has different weight than recent evidence
- Skills can be promoted (learning → proficient) but also demoted (active → inactive)
- Projects move through states: planned → active → delivered → archived

### Source Precedence
1. Explicit user statement or correction (highest)
2. User-authored documentation (PRDs, notes)
3. Career-ladder expectations (baseline)
4. Repeated behavior across conversations
5. Single inferred signal (lowest)

---

## Storage

- **Checkpoints**: `ai_fde_hackathon_catalog.automatic_user_context_profiles.profile_checkpoints_v2`
- **Final profiles**: `ai_fde_hackathon_catalog.automatic_user_context_profiles.user_context_profiles`
- **Model used**: `databricks-claude-opus-4-8`

---

## Updating the Profile

To process new conversations or documents:
1. Load the latest Combo C checkpoint from `profile_checkpoints_v2`
2. Process new evidence chronologically using `update_profiles()`
3. Save updated checkpoint

The checkpoint system ensures no work is lost on session disconnection.
