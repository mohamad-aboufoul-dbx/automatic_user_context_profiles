# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Configuration (YAML-driven)
# Automatic User Context Profile Builder
# =========================================
# ITERATIVE ARCHITECTURE: Profiles are built incrementally, processing one
# piece of evidence at a time in chronological order. This preserves temporal
# signal ("asked about X once" ≠ "expert in X") and avoids batch-inference errors.
#
# CONFIGURATION:
#   Set CONFIG_PATH below to point to a user YAML config file in configs/.
#   Each user creates their own YAML (e.g., configs/mohamad_aboufoul.yaml).
#   See configs/ folder for the template and examples.
#
# DATA SOURCE COMBINATIONS:
#   A: Cold-start only (career ladder) → initial profile set
#   B: A + conversations processed one-at-a-time chronologically
#   C: B + documents processed chunk-by-chunk with date awareness
#
# PROFILE LEVELS (L0 is generated once; L1-L3 are updated iteratively):
#   Level 0 - Cold-Start: Identity + role baseline (never changes after init)
#   Level 1 - Compact Runtime: Communication prefs, active threads, expertise
#   Level 2 - Detailed Engagement: Project threads, capability model, growth
#   Level 3 - Evidence Records: Atomic observations with provenance

import yaml
from datetime import date

# =============================================================================
# SET YOUR CONFIG PATH HERE (or pass via notebook widget)
# =============================================================================
CONFIG_PATH = "/Workspace/Users/mohamad.aboufoul@databricks.com/automatic_user_context_profiles/configs/mohamad_aboufoul.yaml"

# --- Load configuration from YAML ---
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# --- Derived configuration variables ---
# User identity
USER_NAME = config["user"]["name"]
USER_TITLE = config["user"]["title"]
USER_LEVEL = config["user"]["level"]
USER_TENURE = config["user"]["tenure"]
USER_TEAM = config["user"]["team"]

# Data locations
CATALOG = config["data"]["catalog"]
SCHEMA = config["data"]["schema"]
CHAT_TABLE = f"{CATALOG}.{SCHEMA}.{config['data']['chat_table']}"
COLD_START_PATH = config["data"]["cold_start_volume"]
CAREER_LADDER_FILE = config["data"]["career_ladder_file"]
ADDITIONAL_DOCS_PATH = config["data"]["additional_docs_volume"]

# Career ladder level mapping
LEVEL_COLUMNS = config["career_ladder"]["level_columns"]
CAREER_LADDER_CUMULATIVE = config["career_ladder"].get("cumulative", True)

# Model
DEFAULT_MODEL = config["model"]["name"]

# Processing options
CHUNK_SIZE = config["options"].get("chunk_size", 40000)
MAX_RESPONSE_LENGTH = config["options"].get("max_response_length", 1500)

# Runtime
CURRENT_DATE = str(date.today())

# Checkpoint table (shared across all users)
CHECKPOINT_TABLE = f"{CATALOG}.{SCHEMA}.profile_checkpoints_v2"

print(f"Config loaded from: {CONFIG_PATH}")
print(f"User: {USER_NAME} | {USER_TITLE} (Level {USER_LEVEL})")
print(f"Chat table: {CHAT_TABLE}")
print(f"Cold-start: {COLD_START_PATH}{CAREER_LADDER_FILE}")
print(f"Additional docs: {ADDITIONAL_DOCS_PATH}")
print(f"Model: {DEFAULT_MODEL}")
print(f"Level column: {LEVEL_COLUMNS.get(USER_LEVEL, 'NOT FOUND')}")
print(f"Chunk size: {CHUNK_SIZE:,} chars")
print(f"Current date: {CURRENT_DATE}")
print(f"Architecture: Iterative (chronological evidence processing)")

# COMMAND ----------

# DBTITLE 1,Step 1: Extract Cold-Start Data (Career Ladder)
# Extract and format the career ladder for the configured user level.
# Supports any level defined in the config's career_ladder.level_columns mapping.

cold_start_df = spark.sql(f"""
  SELECT * FROM read_files(
    '{COLD_START_PATH}{CAREER_LADDER_FILE}',
    format => 'excel',
    headerRows => 1
  )
""")

rows = cold_start_df.collect()

# --- Dynamic level column resolution ---
# Look up the column header for the configured level
target_level_col = LEVEL_COLUMNS.get(USER_LEVEL)
if not target_level_col:
    available = list(LEVEL_COLUMNS.keys())
    raise ValueError(f"Level '{USER_LEVEL}' not found in career_ladder.level_columns. Available: {available}")

# Verify the column exists in the data
available_cols = cold_start_df.columns
if target_level_col not in available_cols:
    print(f"WARNING: Column '{target_level_col}' not found. Available columns:")
    for c in available_cols:
        print(f"  - {repr(c)}")
    raise ValueError(f"Column '{target_level_col}' not in Excel. Check career_ladder.level_columns in your config.")

# --- Build career ladder text ---
career_ladder_text = f"""AI FDE Career Ladder - Level {USER_LEVEL} ({USER_TITLE})
{'='*60}
"""

if CAREER_LADDER_CUMULATIVE:
    career_ladder_text += f"Note: {USER_LEVEL} expectations are cumulative (includes all lower levels).\n\n"

    # Include all levels up to and including the target level
    level_order = list(LEVEL_COLUMNS.keys())  # assumes ordered L3, L4, L5, L6...
    target_idx = level_order.index(USER_LEVEL)
    levels_to_include = level_order[:target_idx + 1]

    for level in levels_to_include:
        col = LEVEL_COLUMNS[level]
        if col in available_cols:
            career_ladder_text += f"\n{'='*40}\n{level} Expectations:\n{'='*40}\n"
            for row in rows:
                category = row["_c0"]
                value = row[col]
                if category and value:
                    career_ladder_text += f"\n## {category}\n{value}\n"
                elif category and not value:
                    career_ladder_text += f"\n--- {category} ---\n"
else:
    # Only the target level
    for row in rows:
        category = row["_c0"]
        value = row[target_level_col]
        if category and value:
            career_ladder_text += f"\n## {category}\n{value}\n"
        elif category and not value:
            career_ladder_text += f"\n--- {category} ---\n"

print(f"Career ladder extracted for {USER_LEVEL}: {len(career_ladder_text)} chars")
print(f"\n{career_ladder_text[:2000]}...")

# COMMAND ----------

# DBTITLE 1,Step 2: Extract Chat History
# Extract and consolidate chat history into conversation threads

chat_df = spark.sql(f"""
  SELECT Username, Query, Response, Chat_step, Conversation_id, Tool, Datetime
  FROM {CHAT_TABLE}
  ORDER BY Conversation_id, Chat_step
""")

chat_rows = chat_df.collect()

# Group by conversation
from collections import defaultdict
conversations = defaultdict(list)
for row in chat_rows:
    conversations[row["Conversation_id"]].append({
        "step": row["Chat_step"],
        "query": row["Query"],
        "response": row["Response"],
        "tool": row["Tool"],
        "datetime": row["Datetime"]
    })

# Format conversations for profile building
def format_conversation(conv_id, messages):
    tool = messages[0]["tool"]
    dt = messages[0]["datetime"]
    text = f"\n--- Conversation {conv_id} (Tool: {tool}, Date: {dt}) ---\n"
    for msg in messages:
        text += f"\nUser: {msg['query']}\n"
        # Truncate very long responses to keep context manageable
        resp = msg['response']
        if len(resp) > MAX_RESPONSE_LENGTH:
            resp = resp[:MAX_RESPONSE_LENGTH] + "... [truncated]"
        text += f"Assistant: {resp}\n"
    return text

# Build full chat history text
chat_history_text = f"Chat History - {USER_NAME}\n" + "="*50 + "\n"
chat_history_text += f"Total: {len(chat_rows)} messages across {len(conversations)} conversations\n"
chat_history_text += f"Tools used: Glean, Perplexity, Gemini\n"
chat_history_text += f"Date range: June 29 - August 25, 2026\n\n"

for conv_id in sorted(conversations.keys()):
    chat_history_text += format_conversation(conv_id, conversations[conv_id])

print(f"Chat history consolidated: {len(chat_history_text)} chars, {len(conversations)} conversations")
print(f"\nFirst conversation preview:")
print(chat_history_text[:1500] + "...")

# COMMAND ----------

# DBTITLE 1,Step 3: Extract Additional Context Documents
# Parse additional context PDFs using ai_parse_document
# Use SQL to extract text content from VARIANT (most reliable approach)

additional_docs_df = spark.sql(f"""
  WITH parsed AS (
    SELECT 
      path,
      ai_parse_document(content, MAP('version', '2.0')) AS parsed_content
    FROM read_files('{ADDITIONAL_DOCS_PATH}', format => 'binaryFile')
  )
  SELECT
    path,
    concat_ws('\\n\\n',
      transform(
        try_cast(parsed_content:document:elements AS ARRAY<VARIANT>),
        element -> try_cast(element:content AS STRING)
      )
    ) AS full_text
  FROM parsed
  WHERE is_variant_null(parsed_content:error_status)
""")

additional_docs_rows = additional_docs_df.collect()

# Store extracted text by filename
additional_docs_texts = {}
for row in additional_docs_rows:
    path = row["path"]
    filename = path.split("/")[-1]
    doc_text = row["full_text"] or ""
    additional_docs_texts[filename] = doc_text
    print(f"Parsed: {filename} -> {len(doc_text)} chars")

print(f"\nTotal additional documents: {len(additional_docs_texts)}")

# COMMAND ----------

# DBTITLE 1,Step 3b: Prepare Documents with Temporal Metadata
# =============================================================================
# PREPARE DOCUMENTS WITH TEMPORAL METADATA
# =============================================================================
# For iterative processing, we need:
#   1. Date extracted from each document title (for chronological ordering)
#   2. Large docs chunked with date-context preserved per chunk
#   3. Each chunk tagged with its temporal context

import re
from datetime import datetime

def extract_date_from_title(title):
    """Extract date from document title. Returns (date_str, datetime_obj) or (None, None)."""
    # Try patterns: "Month DD, YYYY" or "Month D, YYYY"
    patterns = [
        r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            date_str = match.group(1)
            try:
                for fmt in ["%B %d, %Y", "%B %d %Y", "%m/%d/%Y"]:
                    try:
                        return date_str, datetime.strptime(date_str.replace(",", ""), "%B %d %Y")
                    except ValueError:
                        continue
            except:
                pass
            return date_str, None
    return None, None


def chunk_text(text, max_chars=CHUNK_SIZE):
    """Split text into chunks respecting paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current = ""
    paragraphs = text.split("\n\n")
    
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current += "\n\n" + para if current else para
    
    if current:
        chunks.append(current)
    return chunks


# --- Prepare documents for iterative processing ---
# Each document becomes a list of chunks, each with temporal context.
# Documents are sorted chronologically by title date.

document_queue = []  # list of {name, date_str, date_obj, chunks: [{text, chunk_num}]}

for doc_name, doc_text in additional_docs_texts.items():
    date_str, date_obj = extract_date_from_title(doc_name)
    chunks = chunk_text(doc_text, max_chars=40000)
    
    document_queue.append({
        "name": doc_name,
        "date_str": date_str or "unknown",
        "date_obj": date_obj,
        "total_chars": len(doc_text),
        "chunks": [{"text": c, "chunk_num": i+1, "total_chunks": len(chunks)} for i, c in enumerate(chunks)]
    })

# Sort documents chronologically (unknown dates go last)
document_queue.sort(key=lambda d: d["date_obj"] or datetime(2099, 1, 1))

print("Documents prepared for iterative processing (chronological order):")
print(f"{'-'*70}")
for doc in document_queue:
    print(f"  [{doc['date_str']}] {doc['name']}")
    print(f"    {doc['total_chars']:,} chars -> {len(doc['chunks'])} chunk(s)")
print(f"\nTotal chunks to process: {sum(len(d['chunks']) for d in document_queue)}")

# COMMAND ----------

# DBTITLE 1,Step 4: Profile Level Templates (with Interaction Patterns)
# =============================================================================
# PROFILE LEVEL TEMPLATES
# =============================================================================
# Four profile levels form a stack of increasing detail:
#   Level 0 - Cold-Start: Safe baseline from role/identity metadata
#   Level 1 - Compact Runtime: Quick personalization for everyday turns
#   Level 2 - Detailed Engagement: Full context for planning and deep work
#   Level 3 - Evidence Records: Atomic observations with provenance
#
# Each data-source combo (A, B, C) produces ALL 4 levels as a set.

PROFILE_TEMPLATES = {
    "level_0_cold_start": """
cold_start_profile:
  identity:
    name: ""
    title: ""
    team_or_function: ""
    tenure: ""

  role_context:
    role_summary: ""
    primary_responsibilities: []
    typical_stakeholders: []
    career_level: ""

  expected_capabilities:
    source: "career_ladder"
    technical_acumen: []
    technical_leadership: []
    specialization: []
    project_leadership: []
    customer_relationship_management: []
    customer_engagements: []
    trusted_advisor: []

  initial_context:
    current_projects: "unknown"
    current_objectives: "unknown"
    stated_learning_goals: "unknown"
    stated_response_preferences: "unknown"

  provenance:
    sources: []
    inferred_fields_allowed: false
""",

    "level_1_compact": """
compact_profile:
  about_user:
    name: ""
    title: ""
    tenure: ""
    works_with: []

  current_focus:
    active_projects: []
    this_week_objectives: []
    near_term_objectives: []
    long_term_objectives: []

  capability_snapshot:
    strongest_areas: []
    proficient_in: []
    developing_in: []
    still_learning: []

  interaction_guidance:
    default_response_style: ""
    by_task:
      debugging: ""
      product_questions: ""
      architecture: ""
      project_scoping: ""
      customer_preparation: ""
      learning: ""
    response_pattern_claims:  # distilled from observed interactions
      - claim: ""
        scope_task_types: []
        preferred_structure: []
        avoid: []
        confidence: 0.0

  active_threads: []

  runtime_rules:
    answer_the_underlying_goal: true
    avoid_repeating_known_context: true
    surface_uncertainty_when_profile_is_weak: true
""",

    "level_2_detailed": """
detailed_profile:
  about_me:
    name: ""
    title: ""
    tenure: ""
    team: ""
    works_with:
      internal: []
      customer_or_external: []
    role_summary: ""

  objectives:
    this_week: []
    near_term: []
    long_term: []

  capability_model:
    technical: []
    project_leadership: []
    customer_management: []
    soft_skills: []
    collaboration: []

  communication_model:
    general_preferences: {}
    by_context: {}

  interaction_outcomes:  # Event-level records of what worked/failed
    - outcome_id: ""
      conversation_id: ""
      task_type: ""
      agent_response_summary: ""
      outcome: ""  # worked | did_not_work | partially_worked
      failure_mode: ""  # buried_answer | too_verbose | too_basic | unexplained_jargon | wrong_assumption | insufficient_context | missing_example | wrong_format | premature_solutioning | null
      observable_user_signals: []
      interpretation: ""
      repair_strategy: []
      confidence: 0.0

  response_pattern_claims:  # Distilled, reusable inferences from multiple outcomes
    - claim_id: ""
      category: "response_pattern"
      claim: ""
      scope:
        task_types: []
        projects: []
      supporting_outcome_ids: []
      confidence: 0.0
      runtime_guidance:
        preferred_structure: []
        avoid: []
      status: "active"  # active | tentative | superseded

  project_threads: []

  growth_model:
    actively_leveling_up: []
    still_learning: []
    practice_opportunities: []

  customer_context:
    recurring_customers_or_accounts: []
    customer_domains: []

  profile_metadata:
    last_updated: ""
    sources_used: []
""",

    "level_3_evidence": """
evidence_records:
  - record_id: ""
    kind: ""  # explicit_preference | inferred_preference | topic_expertise | active_goal | project_thread | customer_context | decision | successful_explanation | failed_explanation | interaction_outcome | response_pattern
    attribute: ""  # dot-path into the profile, e.g. communication.preferred_structure
    value: ""
    scope:
      products: []
      task_types: []
      thread_id: null
    status: "active"  # active | superseded | expired | user_deleted
    source:
      type: ""  # conversation_thread | user_feedback | supplementary_document | career_ladder | activity_signal
      reference_id: ""
      excerpt: ""
    inference:
      method: ""  # explicit_statement | repeated_behavior | single_observation | document_extraction | career_ladder_baseline
      confidence: 0.0
      user_confirmed: false
    temporal:
      first_seen: ""
      last_seen: ""
      review_after: ""
      decay_policy: ""  # slow | medium | fast | session_only
    governance:
      visibility: "private_to_user"
      deletable_by_user: true
"""
}

print(f"Profile templates defined: {list(PROFILE_TEMPLATES.keys())}")
print(f"Each data-source combo will produce all 4 levels as a set.")

# COMMAND ----------

# DBTITLE 1,Step 5: Iterative Profile Engine
# =============================================================================
# ITERATIVE PROFILE ENGINE
# =============================================================================
# Two core operations:
#   1. generate_cold_start() - Produces initial L0-L3 from career ladder
#   2. update_profiles() - Takes current L1+L2+L3 + new evidence -> updated L1+L2+L3
#
# L0 is NEVER modified after cold-start (it represents the baseline).
# L1-L3 are updated iteratively as each piece of evidence is processed.

import time
import datetime
import copy

def call_llm(prompt, model=DEFAULT_MODEL):
    """Call the LLM via ai_query. Returns the response text."""
    prompt_df = spark.createDataFrame([(prompt,)], ["prompt_text"])
    prompt_df.createOrReplaceTempView("_profile_prompt")
    result_df = spark.sql(f"""
      SELECT ai_query('{model}', prompt_text) AS response
      FROM _profile_prompt
    """)
    return result_df.collect()[0]["response"]


def generate_cold_start():
    """Generate all 4 profile levels from career ladder only. Returns dict of level->yaml."""
    identity = f"""User Identity:
- Name: {USER_NAME}
- Title: {USER_TITLE} (Level {USER_LEVEL})
- Tenure: {USER_TENURE}
- Team: AI Field Engineering (AI FDE)
- Current Date: {CURRENT_DATE}

Career Ladder (expectations for this level, NOT proven capabilities):
{career_ladder_text}"""

    profiles = {}
    for level_key, template in PROFILE_TEMPLATES.items():
        prompt = f"""You are initializing a user context profile from ONLY career ladder metadata.
This is a COLD-START — you have NO conversation history, NO project documents, NO behavioral evidence.

Data:
{identity}

Generate the {level_key.replace('_', ' ')} profile.
Rules:
- The career ladder describes EXPECTED scope for this level, NOT proven individual capabilities
- Mark all current_projects, objectives, active_threads, and preferences as 'unknown'
- Do NOT infer personality, communication style, or active work
- For evidence records (L3): only include career_ladder_baseline records with confidence ~0.5
- Be honest about what you don't know

Output valid YAML following this structure (no markdown fences):
{template}"""
        profiles[level_key] = call_llm(prompt)
    return profiles


def update_profiles(current_profiles, evidence_text, evidence_date, evidence_type, evidence_source_id):
    """Update L1, L2, L3 given new evidence. L0 is never changed.
    
    Args:
        current_profiles: dict with keys level_1_compact, level_2_detailed, level_3_evidence
        evidence_text: the new conversation or document chunk
        evidence_date: string date when this evidence is from
        evidence_type: 'conversation' or 'document_chunk'
        evidence_source_id: identifier for the source (conv_id or doc_name)
    
    Returns:
        Updated dict with the same keys (L1, L2, L3 updated)
    """
    prompt = f"""You are maintaining a user context profile by processing new evidence chronologically.
The current date is {CURRENT_DATE}. This evidence is from: {evidence_date}.

CURRENT PROFILE STATE:

--- Level 1 (Compact Runtime) ---
{current_profiles['level_1_compact']}

--- Level 2 (Detailed Engagement) ---
{current_profiles['level_2_detailed']}

--- Level 3 (Evidence Records) ---
{current_profiles['level_3_evidence']}

---

NEW EVIDENCE ({evidence_type}, source: {evidence_source_id}, date: {evidence_date}):
{evidence_text}

---

INSTRUCTIONS FOR UPDATING:
1. Review the new evidence and determine what profile updates are warranted
2. Be CONSERVATIVE with capability claims:
   - One question about a topic = 'still_learning' at most, NOT 'proficient_in'
   - Only promote to 'proficient_in' or 'strongest_areas' if MULTIPLE pieces of evidence show applied expertise
   - Asking basic questions about X ≠ working on X ≠ expert in X
3. Track chronological progression:
   - If previous evidence showed "still_learning: X" and new evidence shows the user applying X successfully, THEN consider promoting
   - Update first_seen/last_seen dates in evidence records
4. For communication style: infer ONLY from explicit user feedback or repeated patterns across conversations
5. For projects: only add to active_projects if the evidence shows actual work being done, not just questions asked
6. Add new evidence records (L3) for any notable observations from this evidence
7. If nothing in this evidence warrants a profile change, return the profiles UNCHANGED
8. RESPONSE OUTCOMES AND INTERACTION PATTERNS (critical for conversations):
   - Look for signs of FAILED interactions: user corrections, simplification requests, "that's not what I meant", re-asking the same question differently, abandoning a thread, expressing frustration
   - Look for signs of SUCCESSFUL interactions: user thanks, builds on the answer, moves to next step, applies the suggestion
   - Record interaction_outcomes in Level 2 with: failure_mode (buried_answer, too_verbose, too_basic, unexplained_jargon, wrong_assumption, insufficient_context, missing_example, wrong_format, premature_solutioning), observable_user_signals, and repair_strategy
   - After 2+ similar outcomes, distill a response_pattern_claim (e.g., "For debugging tasks, lead with the answer then provide evidence")
   - Track what WORKED (successful_explanation) as carefully as what failed
   - A single failure is a scoped observation, NOT a permanent global preference change
   - From these patterns, infer interaction_guidance preferences (L1) and repair strategies (L2)

Return ALL THREE levels as a single YAML document with this structure (no markdown fences):

updated_profiles:
  level_1_compact:
    [... complete updated Level 1 ...]
  level_2_detailed:
    [... complete updated Level 2 ...]
  level_3_evidence:
    [... complete updated Level 3 with any new records appended ...]
"""
    response = call_llm(prompt)
    
    # Parse the response - extract each level section
    # Simple split approach: look for the level markers
    updated = {}
    
    # Try to find the sections in the response
    l1_marker = "level_1_compact:"
    l2_marker = "level_2_detailed:"
    l3_marker = "level_3_evidence:"
    
    l1_start = response.find(l1_marker)
    l2_start = response.find(l2_marker)
    l3_start = response.find(l3_marker)
    
    if l1_start >= 0 and l2_start > l1_start and l3_start > l2_start:
        updated['level_1_compact'] = response[l1_start:l2_start].strip()
        updated['level_2_detailed'] = response[l2_start:l3_start].strip()
        updated['level_3_evidence'] = response[l3_start:].strip()
    else:
        # Fallback: return the whole response as-is for each level
        # This handles cases where the model returns a different format
        updated['level_1_compact'] = current_profiles['level_1_compact']
        updated['level_2_detailed'] = current_profiles['level_2_detailed']
        updated['level_3_evidence'] = response  # At minimum capture new evidence
    
    return updated


print("Iterative profile engine loaded.")
print("  - generate_cold_start(): Produces initial L0-L3 from career ladder")
print("  - update_profiles(): Updates L1-L3 from one piece of evidence")
print(f"  - Model: {DEFAULT_MODEL}")

# COMMAND ----------

# DBTITLE 1,Step 6: Combo A — Cold-Start + Checkpoint
# =============================================================================
# COMBO A: COLD-START PROFILE GENERATION + PER-STEP CHECKPOINT
# =============================================================================
# Generates initial L0-L3 from career ladder, saves checkpoint immediately.
# Checkpoint table stores the latest profile state at each processing step,
# so any session loss resumes from the last completed step.

# v2: includes interaction_outcomes and response_pattern_claims from Ontology PDF
# CHECKPOINT_TABLE is defined in Cell 1 (from config)

def save_checkpoint(combo_id, combo_label, profiles_dict, step_id, data_sources):
    """Save current profile state as a checkpoint. Overwrites previous state for this combo."""
    run_ts = datetime.datetime.now().isoformat()
    records = [
        (USER_NAME, combo_id, combo_label, lk, DEFAULT_MODEL,
         data_sources, yaml, len(yaml), step_id, run_ts)
        for lk, yaml in profiles_dict.items()
    ]
    ckpt_df = spark.createDataFrame(records,
        ["user_name", "data_combo", "data_combo_label", "profile_level",
         "model_used", "data_sources", "profile_yaml", "profile_length_chars",
         "step_id", "generated_at"])
    
    # Try delete+append; if table schema is incompatible, overwrite entire table
    try:
        spark.sql(f"DELETE FROM {CHECKPOINT_TABLE} WHERE data_combo = '{combo_id}' AND user_name = '{USER_NAME}'")
        ckpt_df.write.mode("append").saveAsTable(CHECKPOINT_TABLE)
    except Exception as e:
        print(f"  (Schema mismatch detected, recreating checkpoint table)")
        # Load any other combos' data first
        try:
            other_data = spark.sql(f"SELECT * FROM {CHECKPOINT_TABLE} WHERE data_combo != '{combo_id}'")
            combined = other_data.unionByName(ckpt_df, allowMissingColumns=True)
            combined.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CHECKPOINT_TABLE)
        except:
            # Table doesn't exist or is completely broken — just overwrite
            ckpt_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CHECKPOINT_TABLE)


def load_checkpoint(combo_id):
    """Load checkpoint for a combo. Returns (profiles_dict, step_id) or (None, None)."""
    try:
        rows = spark.sql(f"""
            SELECT profile_level, profile_yaml, step_id FROM {CHECKPOINT_TABLE}
            WHERE data_combo = '{combo_id}' AND user_name = '{USER_NAME}'
        """).collect()
        if rows:
            profiles = {row['profile_level']: row['profile_yaml'] for row in rows}
            step_id = rows[0]['step_id']  # same for all rows in a combo
            return profiles, step_id
    except:
        pass
    return None, None


def ensure_checkpoint_table():
    """Create checkpoint table if it doesn't exist (preserves other users' data)."""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {CHECKPOINT_TABLE} (
            user_name STRING,
            data_combo STRING,
            data_combo_label STRING,
            profile_level STRING,
            model_used STRING,
            data_sources STRING,
            profile_yaml STRING,
            profile_length_chars LONG,
            step_id STRING,
            generated_at STRING
        )
    """)
    print(f"Checkpoint table ready: {CHECKPOINT_TABLE}")

ensure_checkpoint_table()

# --- Combo A: Check for existing checkpoint ---
combo_a_profiles, a_step = load_checkpoint("A")

if combo_a_profiles:
    print(f"RESUMED from checkpoint: Combo A (step: {a_step})")
    for lk, yaml_text in combo_a_profiles.items():
        print(f"  {lk}: {len(yaml_text)} chars")
else:
    print("Generating cold-start profiles (Combo A)...")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"{'='*70}")
    
    start_total = time.time()
    combo_a_profiles = generate_cold_start()
    
    for lk, yaml_text in combo_a_profiles.items():
        print(f"  {lk}: {len(yaml_text)} chars")
    
    print(f"{'='*70}")
    print(f"Cold-start profiles generated in {time.time() - start_total:.1f}s")
    
    # Checkpoint immediately
    save_checkpoint("A", "Cold-start only", combo_a_profiles, "cold_start_done", "career_ladder")
    print(f"Checkpoint saved.")

combo_a_final = dict(combo_a_profiles)
print(f"\nCombo A complete. L0 is now fixed. L1-L3 will be updated iteratively.")

# COMMAND ----------

# DBTITLE 1,Step 7: Combo B — Iterative Conversations + Checkpoint
# =============================================================================
# COMBO B: ITERATIVE CONVERSATION PROCESSING + CHECKPOINT
# =============================================================================
# Process each conversation one-at-a-time in chronological order.
# Checkpoint saves at the END of this cell. If it crashes mid-run, re-run
# this cell from scratch (Combo A checkpoint is safe). On re-run, if Combo B
# checkpoint already exists, it resumes instantly.

from datetime import datetime as dt

# --- Resume: if Combo B checkpoint exists, load it and skip ---
combo_b_final, b_step = load_checkpoint("B")

if combo_b_final:
    print(f"RESUMED from checkpoint: Combo B already complete (step: {b_step})")
    for lk, yaml_text in combo_b_final.items():
        print(f"  {lk}: {len(yaml_text)} chars")
else:
    # --- Generate fresh ---
    def parse_conv_datetime(conv_turns):
        """Parse datetime from the first turn of a conversation."""
        date_str = conv_turns[0]["datetime"]
        try:
            return dt.strptime(date_str, "%m/%d/%Y %H:%M:%S")
        except:
            try:
                return dt.strptime(date_str, "%m/%d/%Y %H:%M")
            except:
                return dt(2099, 1, 1)

    conv_list = []
    for conv_id, turns in conversations.items():
        conv_dt = parse_conv_datetime(turns)
        conv_list.append((conv_id, conv_dt, turns))
    conv_list.sort(key=lambda x: x[1])

    print(f"Processing {len(conv_list)} conversations chronologically...")
    print(f"Date range: {conv_list[0][1].strftime('%Y-%m-%d')} to {conv_list[-1][1].strftime('%Y-%m-%d')}")
    print(f"{'='*70}")

    # Start from cold-start state (L1-L3)
    current_profiles = {
        'level_1_compact': combo_a_profiles['level_1_compact'],
        'level_2_detailed': combo_a_profiles['level_2_detailed'],
        'level_3_evidence': combo_a_profiles['level_3_evidence'],
    }

    for i, (conv_id, conv_dt, turns) in enumerate(conv_list):
        tool = turns[0]["tool"]
        date_str = conv_dt.strftime("%Y-%m-%d")
        
        conv_text = f"Conversation with {tool} ({date_str}):\n"
        for turn in turns:
            conv_text += f"\nUser: {turn['query']}\n"
            resp = turn['response'] or ""
            if len(resp) > 1500:
                resp = resp[:1500] + "... [truncated]"
            conv_text += f"Assistant: {resp}\n"
        
        if len(conv_text) > 25000:
            conv_text = conv_text[:25000] + "\n[... conversation truncated ...]"
        
        print(f"  [{i+1}/{len(conv_list)}] Conv {conv_id} ({tool}, {date_str}, {len(turns)} turns)...", end=" ")
        start = time.time()
        
        current_profiles = update_profiles(
            current_profiles=current_profiles,
            evidence_text=conv_text,
            evidence_date=date_str,
            evidence_type="conversation",
            evidence_source_id=f"conv_{conv_id}"
        )
        
        duration = round(time.time() - start, 1)
        total_chars = sum(len(v) for v in current_profiles.values())
        print(f"done ({total_chars:,} chars total, {duration}s)")

    # Assemble Combo B result (L0 never changes)
    combo_b_final = {
        'level_0_cold_start': combo_a_profiles['level_0_cold_start'],
        **current_profiles
    }

    # --- CHECKPOINT: Save at end of cell ---
    save_checkpoint("B", "Cold-start + Chat (iterative)", combo_b_final,
                    f"conv_done_all_{len(conv_list)}", "career_ladder+chat_history_iterative")
    
    print(f"\n{'='*70}")
    print(f"Combo B complete. {len(conv_list)} conversations processed. Checkpoint saved.")
    for lk, yaml_text in combo_b_final.items():
        print(f"  {lk}: {len(yaml_text)} chars")

# COMMAND ----------

# DBTITLE 1,Step 8: Combo C — Iterative Documents + Checkpoint
# =============================================================================
# COMBO C: ITERATIVE DOCUMENT PROCESSING + CHECKPOINT
# =============================================================================
# Starting from Combo B's final state, process each document chunk chronologically.
# Checkpoint saves at the END of this cell. If it crashes mid-run, re-run this
# cell (Combo B checkpoint is safe). On re-run, if Combo C checkpoint exists,
# it resumes instantly.

# --- Resume: if Combo C checkpoint exists, load it and skip ---
combo_c_final, c_step = load_checkpoint("C")

if combo_c_final:
    print(f"RESUMED from checkpoint: Combo C already complete (step: {c_step})")
    for lk, yaml_text in combo_c_final.items():
        print(f"  {lk}: {len(yaml_text)} chars")
else:
    # --- Generate fresh ---
    total_chunks = sum(len(d['chunks']) for d in document_queue)
    print(f"Processing {len(document_queue)} documents ({total_chunks} total chunks)...")
    print(f"{'='*70}")

    # Start from Combo B state
    current_profiles = {
        'level_1_compact': combo_b_final['level_1_compact'],
        'level_2_detailed': combo_b_final['level_2_detailed'],
        'level_3_evidence': combo_b_final['level_3_evidence'],
    }

    chunk_counter = 0
    for doc in document_queue:
        doc_name = doc['name']
        doc_date = doc['date_str']
        num_chunks = len(doc['chunks'])
        
        print(f"\n  Document: {doc_name}")
        print(f"  Date: {doc_date} | Chunks: {num_chunks}")
        
        for chunk_info in doc['chunks']:
            chunk_counter += 1
            chunk_num = chunk_info['chunk_num']
            chunk_text_content = chunk_info['text']
            
            if len(chunk_text_content) > 35000:
                chunk_text_content = chunk_text_content[:35000] + "\n[... chunk truncated ...]"
            
            evidence_text = f"Document: {doc_name}\nDocument date: {doc_date}\nChunk {chunk_num}/{num_chunks}\n\n{chunk_text_content}"
            
            print(f"    Chunk {chunk_num}/{num_chunks} ({len(chunk_text_content):,} chars)...", end=" ")
            start = time.time()
            
            current_profiles = update_profiles(
                current_profiles=current_profiles,
                evidence_text=evidence_text,
                evidence_date=doc_date,
                evidence_type="document_chunk",
                evidence_source_id=f"{doc_name}:chunk_{chunk_num}"
            )
            
            duration = round(time.time() - start, 1)
            print(f"done ({duration}s) [{chunk_counter}/{total_chunks}]")

    # Assemble Combo C result (L0 never changes)
    combo_c_final = {
        'level_0_cold_start': combo_a_profiles['level_0_cold_start'],
        **current_profiles
    }

    # --- CHECKPOINT: Save at end of cell ---
    save_checkpoint("C", "Cold-start + Chat + Docs (iterative)", combo_c_final,
                    f"docs_done_all_{total_chunks}", "career_ladder+chat_history+documents_iterative")
    
    print(f"\n{'='*70}")
    print(f"Combo C complete. {total_chunks} document chunks processed. Checkpoint saved.")
    for lk, yaml_text in combo_c_final.items():
        print(f"  {lk}: {len(yaml_text)} chars")

# COMMAND ----------

# DBTITLE 1,Step 9: Publish Final Profiles from Checkpoints
# =============================================================================
# PUBLISH FINAL PROFILES FROM CHECKPOINTS
# =============================================================================
# Copies checkpoint data to the final `user_context_profiles` table.
# This cell is safe to re-run at any time — it just reads from checkpoints.

FINAL_TABLE = f"{CATALOG}.{SCHEMA}.user_context_profiles"

# Read all checkpoints
all_checkpoints = spark.sql(f"SELECT * FROM {CHECKPOINT_TABLE}")
combo_count = all_checkpoints.select("data_combo").distinct().count()
print(f"Checkpoints found: {all_checkpoints.count()} profiles across {combo_count} combos")

if combo_count < 3:
    print(f"\nWARNING: Only {combo_count}/3 combos have completed. Run remaining combo cells first.")
    print("Publishing what's available...")

# Write to final table
all_checkpoints.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(FINAL_TABLE)

print(f"\nPublished to {FINAL_TABLE}")
print(f"  Total profiles: {all_checkpoints.count()}")
print(f"  Model: {DEFAULT_MODEL}")
print(f"  Architecture: Iterative (chronological evidence processing)")

# Size summary
for combo_id in ['A', 'B', 'C']:
    combo_rows = all_checkpoints.filter(f"data_combo = '{combo_id}'").collect()
    if combo_rows:
        total = sum(row['profile_length_chars'] for row in combo_rows)
        print(f"  Combo {combo_id}: {total:,} chars across {len(combo_rows)} levels")

# COMMAND ----------

# DBTITLE 1,Step 10: Preview & Compare
# =============================================================================
# PREVIEW & COMPARE: Show how profiles evolved across combos
# =============================================================================
# Reads from checkpoints table — works even after session restart.

preview_data = spark.sql(f"""
    SELECT data_combo, data_combo_label, profile_level, profile_yaml, profile_length_chars
    FROM {CHECKPOINT_TABLE}
    ORDER BY data_combo, profile_level
""").collect()

# Organize into dict
combo_profiles = {}
for row in preview_data:
    combo_id = row['data_combo']
    if combo_id not in combo_profiles:
        combo_profiles[combo_id] = {'label': row['data_combo_label'], 'profiles': {}}
    combo_profiles[combo_id]['profiles'][row['profile_level']] = row['profile_yaml']

print("LEVEL 1 (COMPACT) COMPARISON ACROSS COMBOS")
print(f"{'='*70}")
for combo_id in ['A', 'B', 'C']:
    if combo_id in combo_profiles:
        info = combo_profiles[combo_id]
        print(f"\n--- COMBO {combo_id}: {info['label']} ---")
        l1 = info['profiles'].get('level_1_compact', '')
        print(l1[:1500] + ("\n..." if len(l1) > 1500 else ""))

print(f"\n\n{'='*70}")
print("SIZE SUMMARY (chars per profile level):")
print(f"{'='*70}")
levels = ['level_0_cold_start', 'level_1_compact', 'level_2_detailed', 'level_3_evidence']
print(f"{'Level':<25} {'Combo A':<12} {'Combo B':<12} {'Combo C':<12}")
print(f"{'-'*61}")
for lk in levels:
    a = len(combo_profiles.get('A', {}).get('profiles', {}).get(lk, ''))
    b = len(combo_profiles.get('B', {}).get('profiles', {}).get(lk, ''))
    c = len(combo_profiles.get('C', {}).get('profiles', {}).get(lk, ''))
    print(f"{lk:<25} {a:<12} {b:<12} {c:<12}")

print(f"\nProfiles grow incrementally: A (baseline) < B (+conversations) < C (+documents)")

# COMMAND ----------

# DBTITLE 1,Summary & Next Steps
# MAGIC %md
# MAGIC ## Architecture: Iterative Profile Builder
# MAGIC
# MAGIC This notebook builds **Automatic User Context Profiles** using an **iterative, chronological** approach:
# MAGIC
# MAGIC ### Processing Flow
# MAGIC
# MAGIC ```
# MAGIC Career Ladder → Cold-Start Profiles (L0-L3)  [Combo A]
# MAGIC     ↓
# MAGIC For each conversation (chronological): update L1-L3  [Combo B]
# MAGIC     ↓
# MAGIC For each document chunk (chronological): update L1-L3  [Combo C]
# MAGIC ```
# MAGIC
# MAGIC ### Why Iterative?
# MAGIC
# MAGIC - **Temporal signal preserved**: "Asked about X once" ≠ "Expert in X"
# MAGIC - **Conservative capability claims**: Promotes expertise only on repeated evidence
# MAGIC - **Chronological progression**: Tracks what was learned when
# MAGIC - **No context window limits**: Each update is a focused, manageable prompt
# MAGIC
# MAGIC ### Profile Levels
# MAGIC
# MAGIC | Level | Name | Updated? | Runtime Usage |
# MAGIC |-------|------|----------|---------------|
# MAGIC | 0 | Cold-Start | Never (baseline) | Only before any L1+ exists |
# MAGIC | 1 | Compact | After each evidence | **Always** injected in system prompt |
# MAGIC | 2 | Detailed | After each evidence | **On-demand** retrieval by task type |
# MAGIC | 3 | Evidence | After each evidence | **Never** in user convos (maintenance only) |
# MAGIC
# MAGIC ### Data Source Combos
# MAGIC
# MAGIC | Combo | Processing | Result |
# MAGIC |-------|-----------|--------|
# MAGIC | A | Career ladder → one-shot cold start | Baseline profiles |
# MAGIC | B | A + 15 conversations iteratively | Behavioral + interest signal |
# MAGIC | C | B + 4 documents (~28 chunks) iteratively | Full project context |
# MAGIC
# MAGIC ### Key Design Decisions
# MAGIC
# MAGIC - **Model**: `databricks-claude-opus-5` (strongest reasoning for nuanced inference)
# MAGIC - **Conservative updates**: One question about a topic ≠ expertise; requires repeated evidence
# MAGIC - **Profile format**: YAML (structured, human-readable, agent-consumable)
# MAGIC - **L0 immutability**: Cold-start baseline never changes — shows how profiles grow from nothing
# MAGIC
# MAGIC ### Runtime Usage (in new conversations)
# MAGIC
# MAGIC 1. **Level 1** (compact, ~2-3K chars): Always in system prompt — cheap, always-on personalization
# MAGIC 2. **Level 2** (detailed, ~5K chars): Retrieved selectively when task matches a known thread
# MAGIC 3. **Level 3** (evidence records): Used by the profile builder only, never exposed to end-user agents