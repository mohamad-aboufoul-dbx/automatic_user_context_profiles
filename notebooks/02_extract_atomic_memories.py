# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 — Extract Atomic Memories → atomic_memories
# MAGIC
# MAGIC LIVE memory extraction. Re-derives sessions from the raw table exactly as
# MAGIC `01_sessionize` does (01 only makes a temp view, which does NOT persist to a
# MAGIC separate serverless job), applies the reviewed contamination guards from
# MAGIC `src/extract/*`, calls an LLM to extract durable memories per session, and
# MAGIC (in full mode) MERGEs them into the shared `atomic_memories` UC table.
# MAGIC
# MAGIC **This notebook MUST NOT run automatically.** It defaults to `dry_run="true"`
# MAGIC (safe: one LLM call, writes nothing). The controller flips `dry_run="false"`.
# MAGIC
# MAGIC ### Contamination guards (applied BEFORE any LLM call)
# MAGIC   - temporal cutoff `cfg["cutoff_ts"]` (strict `<`)
# MAGIC   - `cfg["heldout_conversation_ids"]` (tight per-task held-out set)
# MAGIC   - the broader cluster ids from `config/heldout_exclusions.json`
# MAGIC
# MAGIC ### Staging (agreed with `scripts/run_extract.sh`)
# MAGIC   - guard modules → `/Volumes/ai_fde_hackathon_catalog/automatic_user_context_profiles/raw/extract_src/extract/*.py`
# MAGIC     (parent dir `.../raw/extract_src` is added to `sys.path`, imported as `extract.*`)
# MAGIC   - config JSONs → `/Volumes/ai_fde_hackathon_catalog/automatic_user_context_profiles/raw/config/*.json`
# MAGIC
# MAGIC ### Sentinel
# MAGIC The runner validates `dbutils.notebook.exit` via `scripts/_check_sentinel.py`,
# MAGIC which reads the `"sentinel"` field. We therefore emit a `"sentinel"` key
# MAGIC ALONGSIDE the task-specified status payload:
# MAGIC   - dry-run: `{"sentinel":"extract_dryrun:OK","status":"OK","stage":"extract_dryrun",...}`
# MAGIC   - full:    `{"sentinel":"extract:OK","status":"OK","stage":"extract",...}`


# COMMAND ----------


import json
import re
import sys
import uuid
from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType,
    ArrayType, FloatType,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATALOG    = "ai_fde_hackathon_catalog"
SCHEMA     = "automatic_user_context_profiles"
RAW_TBL    = f"{CATALOG}.{SCHEMA}.raw_conversations_abdullah_said"
ATOMIC_TBL = f"{CATALOG}.{SCHEMA}.atomic_memories"

VOLUME_ROOT     = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
EXTRACT_SRC_DIR = f"{VOLUME_ROOT}/extract_src"           # parent of the `extract` package
CONFIG_DIR      = f"{VOLUME_ROOT}/config"                # staged compile_config + exclusions
COMPILE_CONFIG_PATH     = f"{CONFIG_DIR}/compile_config.json"
HELDOUT_EXCLUSIONS_PATH = f"{CONFIG_DIR}/heldout_exclusions.json"

LLM_ENDPOINT     = "databricks-claude-sonnet-4-5"
EXTRACTION_MODEL = LLM_ENDPOINT
MAX_MEMORIES     = 20   # matches the extraction prompt cap / schema _ARRAY_CAP

# ---------------------------------------------------------------------------
# dry_run widget — DEFAULT "true" (safe). Only the literal "false" enables writes.
# ---------------------------------------------------------------------------
try:
    dbutils.widgets.text("dry_run", "true")   # noqa: F821 (Databricks-injected)
    _dry_raw = dbutils.widgets.get("dry_run")  # noqa: F821
except Exception:
    _dry_raw = "true"
DRY_RUN = _dry_raw.strip().lower() != "false"
print(f"dry_run param = {_dry_raw!r}  ->  DRY_RUN={DRY_RUN}")


# COMMAND ----------

# MAGIC %md ## Import the reviewed guard modules from the staged Volume path

# COMMAND ----------

# The run script stages src/extract/ as an importable package under
# EXTRACT_SRC_DIR/extract/*.py. Add the PARENT to sys.path and import extract.*.
if EXTRACT_SRC_DIR not in sys.path:
    sys.path.insert(0, EXTRACT_SRC_DIR)

from extract.prefilter import eligible_sessions, unresolved_prefix_ids  # noqa: E402
from extract.schema import validate_memory_array, validate_memory, MEMORY_TYPES  # noqa: E402
from extract.ids import memory_id                         # noqa: E402

print(f"Imported guards from {EXTRACT_SRC_DIR}")
print(f"MEMORY_TYPES ({len(MEMORY_TYPES)}): {sorted(MEMORY_TYPES)}")


# COMMAND ----------

# MAGIC %md ## Load staged config + build the exclusion set

# COMMAND ----------

with open(COMPILE_CONFIG_PATH, "r", encoding="utf-8") as fh:
    CFG = json.load(fh)
with open(HELDOUT_EXCLUSIONS_PATH, "r", encoding="utf-8") as fh:
    EXCL_DOC = json.load(fh)

# Tight per-task held-out ids (from compile_config).
HELDOUT_IDS = set(CFG.get("heldout_conversation_ids", []))

# Broader cluster exclusions: union every conversation_ids list across the
# cluster objects in heldout_exclusions.json (skip the "_note" string field).
CLUSTER_EXCLUSIONS: set[str] = set()
for key, val in EXCL_DOC.items():
    if isinstance(val, dict) and "conversation_ids" in val:
        CLUSTER_EXCLUSIONS.update(val["conversation_ids"])

# The full set that must NEVER appear in atomic_memories.
BANNED_IDS = HELDOUT_IDS | CLUSTER_EXCLUSIONS

print(f"cutoff_ts            : {CFG.get('cutoff_ts')}")
print(f"heldout ids (tight)  : {len(HELDOUT_IDS)}")
print(f"cluster exclusions   : {len(CLUSTER_EXCLUSIONS)}")
print(f"banned ids (union)   : {len(BANNED_IDS)}")


# COMMAND ----------

# MAGIC %md ## Re-derive sessions from the raw table (identical to 01_sessionize)

# COMMAND ----------

df_raw = spark.table(RAW_TBL)
raw_count = df_raw.count()
print(f"Raw rows loaded: {raw_count}")

# Deterministic sort keys first (event_datetime, chat_step, row_hash), then the
# carried fields — byte-for-byte the same struct ordering as 01_sessionize.
df_sessions = (
    df_raw
    .groupBy("conversation_id")
    .agg(
        F.sort_array(
            F.collect_list(
                F.struct(
                    F.col("event_datetime"),   # sort key 1
                    F.col("chat_step"),        # sort key 2
                    F.col("row_hash"),         # sort key 3 / tie-breaker
                    F.col("username"),
                    F.col("query_text"),
                    F.col("response_text"),
                    F.col("source_tool"),
                    F.col("source_name"),
                    F.col("ingested_at"),
                )
            )
        ).alias("events"),
        F.count("*").alias("event_count"),
        F.min("event_datetime").alias("started_at"),
        F.max("event_datetime").alias("ended_at"),
    )
    .withColumn("username",    F.col("events").getItem(0).getField("username"))
    .withColumn("source_tool", F.col("events").getItem(0).getField("source_tool"))
)

session_count = df_sessions.count()
distinct_conv_ids = df_raw.select("conversation_id").distinct().count()
assert session_count == distinct_conv_ids, (
    f"Session count {session_count} != distinct conversation_id count {distinct_conv_ids}"
)
print(f"Sessions produced: {session_count} (one per conversation_id)")


# COMMAND ----------

# MAGIC %md ## Collect sessions to the driver as a list of dicts

# COMMAND ----------

def _iso(ts) -> str | None:
    """Timestamp -> ISO-8601 string (guard functions parse strings)."""
    return ts.isoformat() if ts is not None else None


sessions: list[dict] = []
for row in df_sessions.collect():
    events = []
    for ev in (row["events"] or []):
        events.append({
            "event_datetime": _iso(ev["event_datetime"]),
            "chat_step":      ev["chat_step"],
            "username":       ev["username"],
            "query_text":     ev["query_text"],
            "response_text":  ev["response_text"],
            "source_tool":    ev["source_tool"],
        })
    sessions.append({
        "conversation_id": row["conversation_id"],
        "username":        row["username"],
        "source_tool":     row["source_tool"],
        "started_at":      _iso(row["started_at"]),
        "ended_at":        _iso(row["ended_at"]),   # read by eligible_sessions
        "_ended_at_dt":    row["ended_at"],          # native datetime for source_datetime
        "event_count":     row["event_count"],
        "events":          events,
    })

print(f"Collected {len(sessions)} sessions to the driver")


# COMMAND ----------

# MAGIC %md ## Apply contamination guards BEFORE any LLM call

# COMMAND ----------

total_sessions = len(sessions)

# Freeze-time invariant: the guards match conversation_id by EXACT equality, so
# a banned id that is only a PREFIX of a real conversation_id (e.g. a truncated
# 8-char id vs. the full UUID) would silently leak through. Fail loudly here
# before any extraction if any banned id strict-prefix-matches a real id.
_real_conv_ids = {s["conversation_id"] for s in sessions}
_unresolved = unresolved_prefix_ids(BANNED_IDS, _real_conv_ids)
if _unresolved:
    raise RuntimeError(
        "CONTAMINATION GUARD MISCONFIGURED: these banned ids are only a PREFIX "
        f"of a real conversation_id (exact-equality guards will miss them): {_unresolved}. "
        "Resolve them to full conversation_ids in config/heldout_exclusions.json."
    )

eligible = eligible_sessions(sessions, CFG, CLUSTER_EXCLUSIONS)

# Breakdown: exclusion drops (id in banned) vs cutoff drops (the remainder that
# eligible_sessions dropped are cutoff drops, since only two guard classes exist).
dropped_by_exclusion = sum(1 for s in sessions if s["conversation_id"] in BANNED_IDS)
dropped_by_cutoff = total_sessions - dropped_by_exclusion - len(eligible)

print(f"Total sessions        : {total_sessions}")
print(f"Dropped by exclusion  : {dropped_by_exclusion}  (heldout+cluster ids)")
print(f"Dropped by cutoff     : {dropped_by_cutoff}  (ended_at >= {CFG.get('cutoff_ts')})")
print(f"ELIGIBLE for extract  : {len(eligible)}")

assert dropped_by_cutoff >= 0, (
    f"Negative cutoff-drop count ({dropped_by_cutoff}) — a session was double-counted; "
    "guard logic changed unexpectedly."
)


# COMMAND ----------

# MAGIC %md ## Extraction prompt (verbatim from prompts/extraction_prompt.md v1)

# COMMAND ----------

# Copied verbatim from the fenced SYSTEM/USER block of prompts/extraction_prompt.md
# (v1). Embedded here so the prompt file need not also be staged to the Volume.
SYSTEM_PROMPT = """You extract durable, reusable MEMORIES about one user (Abdullah) and their projects
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
{"memory_text": "Abdullah works on class-agnostic object detection for IR/thermal imagery and wants bounding boxes for high-objectness regions without classification.", "memory_type": "project_state", "domain": "general_workflow", "evidence": "chat_step 2: USER asked about pretrained models like OWLv2 for IR data, class-agnostic detections", "confidence": 0.9}"""

USER_PROMPT_TEMPLATE = """username={{username}} conversation_id={{conversation_id}} tool={{source_tool}}
started={{started_at}} ended={{ended_at}}
Transcript:
{{transcript}}"""


def build_transcript(session: dict) -> str:
    """Join turns with explicit `chat_step N` markers and USER/ASSISTANT text."""
    lines: list[str] = []
    for ev in session["events"]:
        lines.append(f"--- chat_step {ev.get('chat_step')} ---")
        q = (ev.get("query_text") or "").strip()
        r = (ev.get("response_text") or "").strip()
        if q:
            lines.append(f"USER: {q}")
        if r:
            lines.append(f"ASSISTANT: {r}")
    return "\n".join(lines)


def fill_user_prompt(session: dict, transcript: str) -> str:
    return (
        USER_PROMPT_TEMPLATE
        .replace("{{username}}",        str(session.get("username")))
        .replace("{{conversation_id}}", str(session.get("conversation_id")))
        .replace("{{source_tool}}",     str(session.get("source_tool")))
        .replace("{{started_at}}",      str(session.get("started_at")))
        .replace("{{ended_at}}",        str(session.get("ended_at")))
        .replace("{{transcript}}",      transcript)
    )


# COMMAND ----------

# MAGIC %md ## LLM call + response parsing

# COMMAND ----------

from databricks.sdk import WorkspaceClient                     # noqa: E402
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole  # noqa: E402

_WS = WorkspaceClient()


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Query the chat serving endpoint; return the raw assistant text content."""
    resp = _WS.serving_endpoints.query(
        name=LLM_ENDPOINT,
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=ChatMessageRole.USER,   content=user_prompt),
        ],
    )
    if not resp.choices:
        raise ValueError("serving endpoint returned no choices")
    return resp.choices[0].message.content or ""


def parse_memory_response(raw: str) -> list[dict]:
    """Extract the memory list from a raw model response.

    Handles: bare JSON array, `{"memories":[...]}` object, and markdown-fenced
    variants. Returns [] when nothing parseable is found.
    """
    s = (raw or "").strip()
    if not s:
        return []
    # If a markdown code fence appears ANYWHERE (Claude often wraps JSON in
    # ```json … ```, sometimes after a preamble line), prefer its contents.
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()

    def _coerce(obj):
        if isinstance(obj, dict):
            mem = obj.get("memories", [])
            return mem if isinstance(mem, list) else []
        if isinstance(obj, list):
            return obj
        return []

    try:
        return _coerce(json.loads(s))
    except Exception:
        pass

    # Fallback: pull the outermost JSON object/array substring and retry.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = s.find(open_ch)
        end = s.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            try:
                return _coerce(json.loads(s[start:end + 1]))
            except Exception:
                continue
    return []


def validated_memories(candidates: list[dict]) -> tuple[list[dict], list[str]]:
    """Validate the array, then drop invalid items with a logged reason.

    Returns (kept_items, reasons). Array-level errors (e.g. the >20 cap) are
    reported but do not by themselves discard valid items; per-item invalids are
    dropped individually.
    """
    reasons: list[str] = []
    kept: list[dict] = []

    # validate_memory_array reports the array-level >20-cap violation plus each
    # per-item error prefixed "item[i]:". Surface only the array-level message
    # here; per-item invalids are re-reported below as explicit DROP reasons.
    for msg in validate_memory_array(candidates):
        if not msg.startswith("item["):
            reasons.append(msg)

    for i, item in enumerate(candidates):
        # An item is kept only if it has NO per-item validation errors.
        item_errors = validate_memory(item)
        if item_errors:
            reasons.append(f"DROP item[{i}]: {', '.join(item_errors)}")
            continue
        kept.append(item)

    # Enforce the ≤20 cap deterministically after per-item filtering.
    if len(kept) > MAX_MEMORIES:
        reasons.append(f"trimmed kept items from {len(kept)} to cap {MAX_MEMORIES}")
        kept = kept[:MAX_MEMORIES]

    return kept, reasons


# COMMAND ----------

# MAGIC %md ## extraction_run_id (job run id, or uuid4 fallback)

# COMMAND ----------

def get_extraction_run_id() -> str:
    """Best-effort job run id from the notebook context; uuid4 if unavailable."""
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # noqa: F821
        opt = ctx.tags().get("jobRunId")
        if opt is not None and hasattr(opt, "isDefined") and opt.isDefined():
            return str(opt.get())
        # Some runtimes return a plain value or a Some(...) wrapper.
        if opt is not None:
            val = opt.get() if hasattr(opt, "get") else opt
            if val:
                return str(val)
    except Exception as exc:  # noqa: BLE001
        print(f"    (jobRunId unavailable, using uuid4: {exc})")
    return str(uuid.uuid4())


EXTRACTION_RUN_ID = get_extraction_run_id()
print(f"extraction_run_id = {EXTRACTION_RUN_ID}")


# COMMAND ----------

# MAGIC %md ## DRY-RUN: one session only, print everything, write NOTHING

# COMMAND ----------

if DRY_RUN:
    print("=== DRY RUN — one LLM call, no writes ===")
    # Diagnostics surfaced in the exit payload (serverless has no driver logs via
    # get-run-output, so everything the controller needs must ride in the exit).
    sample_conversation_id = None
    sample_event_count = 0
    transcript_chars = 0
    raw_response = ""
    parsed_count = 0
    validated_count = 0
    validation_errors: list[str] = []

    if not eligible:
        print("No eligible sessions — nothing to sample.")
    else:
        s0 = eligible[0]
        sample_conversation_id = s0["conversation_id"]
        sample_event_count = s0["event_count"]
        transcript = build_transcript(s0)
        transcript_chars = len(transcript)
        print(f"conversation_id : {sample_conversation_id}")
        print(f"event_count     : {sample_event_count}")
        print(f"transcript length (chars): {transcript_chars}")
        user_prompt = fill_user_prompt(s0, transcript)
        raw = call_llm(SYSTEM_PROMPT, user_prompt)
        raw_response = raw or ""
        print("\n=== RAW MODEL RESPONSE ===")
        print(raw_response)
        candidates = parse_memory_response(raw_response)
        parsed_count = len(candidates)
        kept, reasons = validated_memories(candidates)
        validated_count = len(kept)
        validation_errors = reasons
        print(f"\nparsed_count={parsed_count}  validated_count={validated_count}")
        print("\n=== VALIDATION REASONS (dropped/notes) ===")
        for r in reasons:
            print(f"  - {r}")
        print("\n=== PARSED + VALIDATED MEMORIES (JSON) ===")
        print(json.dumps(kept, indent=2, default=str))

    sentinel_payload = {
        "sentinel":          "extract_dryrun:OK",
        "status":            "OK",
        "stage":             "extract_dryrun",
        "eligible":          len(eligible),
        "sample_memories":   validated_count,
        # --- diagnostics (no driver logs on serverless) ---
        "conversation_id":   sample_conversation_id,
        "event_count":       sample_event_count,
        "transcript_chars":  transcript_chars,
        "parsed_count":      parsed_count,
        "validated_count":   validated_count,
        "validation_errors": validation_errors[:5],
        "raw_response":      raw_response[:2500],
    }
    dbutils.notebook.exit(json.dumps(sentinel_payload))  # noqa: F821


# COMMAND ----------

# MAGIC %md ## FULL RUN: extract every eligible session, MERGE, assert zero held-out

# COMMAND ----------

# (Only reached when DRY_RUN is False — dry-run exits above.)
MEM_SCHEMA = StructType([
    StructField("memory_id",         StringType(),           False),
    StructField("username",          StringType(),           False),
    StructField("memory_text",       StringType(),           False),
    StructField("memory_type",       StringType(),           False),
    StructField("domain",            StringType(),           False),
    StructField("source_tool",       StringType(),           False),
    StructField("conversation_id",   StringType(),           False),
    StructField("source_datetime",   TimestampType(),        False),
    StructField("evidence",          StringType(),           False),
    StructField("confidence",        DoubleType(),           False),
    StructField("embedding",         ArrayType(FloatType()), True),
    StructField("embedding_model",   StringType(),           True),
    StructField("extraction_model",  StringType(),           False),
    StructField("extraction_run_id", StringType(),           False),
    StructField("created_at",        TimestampType(),        False),
])


def _clean_domain(val) -> str:
    """Validate the model's domain to a sensible non-empty string; default general_workflow."""
    if isinstance(val, str) and val.strip():
        return val.strip()
    return "general_workflow"


rows: list[tuple] = []
n_eligible = len(eligible)
n_llm_failures = 0
n_sessions_with_memories = 0
created_at = datetime.now(timezone.utc)

for idx, session in enumerate(eligible):
    cid = session["conversation_id"]
    try:
        transcript = build_transcript(session)
        user_prompt = fill_user_prompt(session, transcript)
        raw = call_llm(SYSTEM_PROMPT, user_prompt)
        candidates = parse_memory_response(raw)
        kept, reasons = validated_memories(candidates)
        if reasons:
            print(f"[{idx+1}/{n_eligible}] {cid}: {len(reasons)} validation note(s)")
            for r in reasons:
                print(f"      - {r}")
        if kept:
            n_sessions_with_memories += 1
        for m in kept:
            mtext = m["memory_text"]
            mtype = m["memory_type"]
            rows.append((
                memory_id(cid, mtext, mtype),          # memory_id (deterministic)
                session["username"],                    # username
                mtext,                                  # memory_text
                mtype,                                  # memory_type
                _clean_domain(m.get("domain")),         # domain
                session["source_tool"],                 # source_tool
                cid,                                    # conversation_id
                session["_ended_at_dt"],                # source_datetime = session ended_at
                str(m["evidence"]),                     # evidence
                float(m["confidence"]),                 # confidence
                None,                                   # embedding (NULL)
                None,                                   # embedding_model (NULL)
                EXTRACTION_MODEL,                       # extraction_model
                EXTRACTION_RUN_ID,                      # extraction_run_id
                created_at,                             # created_at
            ))
    except Exception as exc:  # noqa: BLE001 — one bad session must not abort the batch
        n_llm_failures += 1
        print(f"[{idx+1}/{n_eligible}] {cid}: EXTRACTION FAILED — {type(exc).__name__}: {exc}")

# De-duplicate rows by memory_id on the driver so the MERGE source is unique
# (identical memory text+type within a conversation collapses to one id).
_seen: set[str] = set()
_deduped: list[tuple] = []
for r in rows:
    if r[0] in _seen:
        continue
    _seen.add(r[0])
    _deduped.append(r)
print(f"Candidate memory rows: {len(rows)} -> {len(_deduped)} after memory_id de-dup")

memories_written = len(_deduped)

if _deduped:
    df_mem = spark.createDataFrame(_deduped, schema=MEM_SCHEMA)
    df_mem.createOrReplaceTempView("_new_atomic_memories")
    spark.sql(f"""
        MERGE INTO {ATOMIC_TBL} t
        USING _new_atomic_memories s
        ON t.memory_id = s.memory_id
        WHEN NOT MATCHED THEN INSERT *
    """)
    print(f"MERGE complete — {memories_written} unique memory rows upserted (idempotent).")
else:
    print("No memories extracted — nothing to MERGE.")


# COMMAND ----------

# MAGIC %md ## Contamination assertion — ZERO held-out/cluster rows in atomic_memories

# COMMAND ----------

# Build the banned-id set as a temp view and count matches by join (no string
# interpolation of ids into SQL). MUST be exactly 0.
banned_df = spark.createDataFrame(
    [(cid,) for cid in sorted(BANNED_IDS)],
    schema=StructType([StructField("conversation_id", StringType(), False)]),
)
banned_df.createOrReplaceTempView("_banned_conversation_ids")

heldout_rows = spark.sql(f"""
    SELECT COUNT(*) AS n
    FROM {ATOMIC_TBL} m
    JOIN _banned_conversation_ids b
      ON m.conversation_id = b.conversation_id
""").first()["n"]

print(f"Held-out/cluster rows present in atomic_memories: {heldout_rows}")
# Explicit raise (NOT assert): asserts are stripped under python -O/-OO, and this
# is the backstop that protects the shared atomic_memories table.
if heldout_rows != 0:
    raise RuntimeError(
        f"CONTAMINATION: {heldout_rows} atomic_memories rows reference held-out/cluster "
        "conversation_ids. Extraction is invalid — refusing to certify this run."
    )
print("PASS: zero held-out/cluster contamination in atomic_memories.")


# COMMAND ----------

# Sentinel: must be the last thing that executes. The runner validates the
# "sentinel" field via scripts/_check_sentinel.py.
sentinel_payload = {
    "sentinel":         "extract:OK",
    "status":           "OK",
    "stage":            "extract",
    "eligible":         n_eligible,
    "memories_written": memories_written,
    "sessions_with_memories": n_sessions_with_memories,
    "llm_failures":     n_llm_failures,
    "heldout_rows":     heldout_rows,
}
dbutils.notebook.exit(json.dumps(sentinel_payload))  # noqa: F821
