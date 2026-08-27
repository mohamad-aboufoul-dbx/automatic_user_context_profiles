# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Sessionize raw_conversations_abdullah_said
# MAGIC
# MAGIC Groups events by conversation_id, orders by event_datetime then chat_step,
# MAGIC and produces ONE session record per conversation_id.
# MAGIC Preserves /goal text, commands, paths, branches, worktrees, test names,
# MAGIC failures, and corrections per SPEC §4.
# MAGIC
# MAGIC Output: UC view `sessions_abdullah_said` (or a temp view used downstream).

# COMMAND ----------

CATALOG  = "ai_fde_hackathon_catalog"
SCHEMA   = "automatic_user_context_profiles"
RAW_TBL  = f"{CATALOG}.{SCHEMA}.raw_conversations_abdullah_said"

# COMMAND ----------

from pyspark.sql import functions as F, Window

# Load raw
df_raw = spark.table(RAW_TBL)
print(f"Raw rows: {df_raw.count()}")

# COMMAND ----------
# MAGIC %md ## Sessionize: one row per conversation_id

# Order events within each conversation consistently
w = Window.partitionBy("conversation_id").orderBy("event_datetime", "chat_step")

df_ordered = (
    df_raw
    .withColumn("_step_rank", F.rank().over(w))
    .orderBy("conversation_id", "event_datetime", "chat_step")
)

# Aggregate per conversation_id — preserve all key signals
df_sessions = (
    df_ordered.groupBy("conversation_id")
    .agg(
        F.first("username").alias("username"),
        F.first("source_tool").alias("source_tool"),
        F.min("event_datetime").alias("started_at"),
        F.max("event_datetime").alias("ended_at"),
        F.count("*").alias("event_count"),
        # Ordered transcript: concat steps in ranked order
        F.collect_list(
            F.struct(
                F.col("_step_rank").alias("rank"),
                F.col("chat_step").alias("chat_step"),
                F.col("event_datetime").alias("event_datetime"),
                F.col("query_text").alias("query_text"),
                F.col("response_text").alias("response_text"),
            )
        ).alias("events_unordered"),
        # Extract /goal line from first query containing /goal
        F.first(
            F.when(
                F.lower(F.col("query_text")).like("%/goal%"),
                F.col("query_text")
            )
        ).alias("goal_raw"),
    )
)

# Re-sort events within each session by rank
# (collect_list ordering is non-deterministic; we'll sort in Python below)
# For display/downstream we add a sorted_transcript as JSON string

from pyspark.sql.types import ArrayType, StructType, StructField, StringType, LongType, TimestampType
import json

def extract_goal_line(goal_raw: str) -> str:
    """Return the /goal line (first line containing /goal, up to 500 chars)."""
    if not goal_raw:
        return None
    for line in goal_raw.split("\n"):
        if "/goal" in line.lower():
            return line.strip()[:500]
    return goal_raw[:500]

def sort_events(events):
    """Sort event structs by (rank, chat_step)."""
    if not events:
        return []
    return sorted(events, key=lambda e: (e.rank, e.chat_step))

def extract_paths(text: str) -> list[str]:
    """Extract filesystem paths from text."""
    import re
    if not text:
        return []
    return list(set(re.findall(r'/[\w./\-_]+(?:\.py|\.md|\.json|\.yaml|\.toml|\.txt|\.sh)?', text)))

def extract_branches(text: str) -> list[str]:
    """Extract branch patterns (polly/*, dev/*, feat/*, fix/*)."""
    import re
    if not text:
        return []
    return list(set(re.findall(r'\b(?:polly|dev|feat|fix|chore|docs)/[\w.\-]+', text)))

def extract_worktrees(text: str) -> list[str]:
    """Extract worktree paths."""
    import re
    if not text:
        return []
    return list(set(re.findall(r'\.worktrees/[\w.\-]+', text)))

def extract_test_names(text: str) -> list[str]:
    """Extract pytest test function/class names."""
    import re
    if not text:
        return []
    return list(set(re.findall(r'\btest_[\w]+|\bTest[\w]+', text)))

def extract_commands(text: str) -> list[str]:
    """Extract shell commands (pytest, ruff, git, databricks, pip, etc.)."""
    import re
    if not text:
        return []
    patterns = [
        r'pytest[\s\w./\-]+',
        r'ruff\s+check[\s\w./\-]+',
        r'git\s+\w+[\s\w./\-]*',
        r'databricks[\s\w./\-]+',
    ]
    found = []
    for pat in patterns:
        found.extend(re.findall(pat, text))
    return list(set(f[:200] for f in found))


@F.udf(returnType=StringType())
def build_session_json(events_unordered, goal_raw: str) -> str:
    """Produce a compact JSON session record with sorted events and extracted signals."""
    events = sort_events(events_unordered) if events_unordered else []

    goal_line = extract_goal_line(goal_raw)

    all_queries    = " ".join(e.query_text or "" for e in events)
    all_responses  = " ".join(e.response_text or "" for e in events)
    all_text       = all_queries + " " + all_responses

    paths      = extract_paths(all_text)
    branches   = extract_branches(all_text)
    worktrees  = extract_worktrees(all_text)
    test_names = extract_test_names(all_text)
    commands   = extract_commands(all_text)

    steps = [
        {
            "rank":          e.rank,
            "chat_step":     e.chat_step,
            "event_datetime": str(e.event_datetime),
            "query_preview":  (e.query_text or "")[:300],
            "response_preview": (e.response_text or "")[:300],
        }
        for e in events
    ]

    record = {
        "goal":       goal_line,
        "steps":      steps,
        "paths":      sorted(set(paths))[:30],
        "branches":   sorted(set(branches))[:10],
        "worktrees":  sorted(set(worktrees))[:10],
        "test_names": sorted(set(test_names))[:20],
        "commands":   sorted(set(commands))[:15],
    }
    return json.dumps(record, default=str)


df_sessions_enriched = (
    df_sessions
    .withColumn("session_json", build_session_json("events_unordered", "goal_raw"))
    .withColumn(
        "goal_line",
        F.udf(extract_goal_line, StringType())("goal_raw")
    )
    .drop("events_unordered")
    .orderBy("started_at")
)

df_sessions_enriched.createOrReplaceTempView("sessions_abdullah_said")

print(f"Sessions: {df_sessions_enriched.count()}")

# COMMAND ----------
# MAGIC %md ## Verify: no cross-conversation merges

assert df_sessions_enriched.count() == df_raw.select("conversation_id").distinct().count(), \
    "Mismatch: session count != distinct conversation_id count — possible merge!"
print("PASS: one session per conversation_id")

# COMMAND ----------
# MAGIC %md ## Sample session record

sample = spark.sql("""
SELECT conversation_id, username, source_tool,
       started_at, ended_at, event_count, goal_line, session_json
FROM sessions_abdullah_said
WHERE goal_line IS NOT NULL
ORDER BY started_at ASC
LIMIT 1
""").first()

if sample:
    import json
    print(f"\n=== SAMPLE SESSION RECORD ===")
    print(f"conversation_id : {sample.conversation_id}")
    print(f"source_tool     : {sample.source_tool}")
    print(f"started_at      : {sample.started_at}")
    print(f"ended_at        : {sample.ended_at}")
    print(f"event_count     : {sample.event_count}")
    print(f"goal_line       : {sample.goal_line}")
    sj = json.loads(sample.session_json)
    print(f"\nPaths detected  : {sj.get('paths', [])[:5]}")
    print(f"Branches        : {sj.get('branches', [])}")
    print(f"Worktrees       : {sj.get('worktrees', [])}")
    print(f"Test names (5)  : {sj.get('test_names', [])[:5]}")
    print(f"Commands  (5)   : {sj.get('commands', [])[:5]}")
    print(f"\nOrdered steps   :")
    for step in sj.get("steps", []):
        print(f"  rank={step['rank']} step={step['chat_step']} dt={step['event_datetime']}")
        print(f"    query: {step['query_preview'][:120]}")

# COMMAND ----------
# MAGIC %md ## /goal sessions overview

spark.sql("""
SELECT conversation_id, started_at, event_count,
       LEFT(goal_line, 200) AS goal_preview
FROM sessions_abdullah_said
WHERE goal_line IS NOT NULL
ORDER BY started_at
""").show(20, truncate=False)

# COMMAND ----------
# MAGIC %md ## Timeline overview

spark.sql("""
SELECT DATE_TRUNC('month', started_at) AS month,
       COUNT(*) AS sessions,
       SUM(event_count) AS events,
       COUNT(CASE WHEN goal_line IS NOT NULL THEN 1 END) AS goal_sessions
FROM sessions_abdullah_said
GROUP BY 1
ORDER BY 1
""").show(20, truncate=False)
