# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 — Sessionize raw_conversations_abdullah_said
# MAGIC
# MAGIC Pure sessionization: groups events by conversation_id, orders them
# MAGIC deterministically by (event_datetime ASC, chat_step ASC, row_hash ASC),
# MAGIC and produces EXACTLY ONE session record per conversation_id.
# MAGIC
# MAGIC Output columns:
# MAGIC   conversation_id  STRING
# MAGIC   username         STRING   — from first (earliest) event
# MAGIC   source_tool      STRING   — from first (earliest) event
# MAGIC   started_at       TIMESTAMP
# MAGIC   ended_at         TIMESTAMP
# MAGIC   event_count      BIGINT
# MAGIC   events           ARRAY<STRUCT<...>>  — complete, ordered; no truncation
# MAGIC
# MAGIC No extraction, no signal parsing, no UDFs. Extraction belongs in notebook 02.


# COMMAND ----------


from pyspark.sql import functions as F

CATALOG = "ai_fde_hackathon_catalog"
SCHEMA  = "automatic_user_context_profiles"
RAW_TBL = f"{CATALOG}.{SCHEMA}.raw_conversations_abdullah_said"

df_raw = spark.table(RAW_TBL)
raw_count = df_raw.count()
print(f"Raw rows loaded: {raw_count}")


# COMMAND ----------

# MAGIC %md ## Sessionize — one row per conversation_id

# COMMAND ----------

# Build a struct with sort keys first so sort_array is deterministic:
#   primary   : event_datetime  (ascending)
#   secondary : chat_step       (ascending)
#   tie-break : row_hash        (ascending — SHA-256 is unique, always breaks ties)
# All remaining fields follow; they are carried along, not sorted on.
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
                    F.col("query_text"),       # complete — not truncated
                    F.col("response_text"),    # complete — not truncated
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
    # Derive username and source_tool from the first sorted event (deterministic).
    .withColumn("username",    F.col("events").getItem(0).getField("username"))
    .withColumn("source_tool", F.col("events").getItem(0).getField("source_tool"))
)

session_count = df_sessions.count()
print(f"Sessions produced: {session_count}")


# COMMAND ----------

# MAGIC %md ## Verify: exactly one row per conversation_id

# COMMAND ----------

distinct_conv_ids = df_raw.select("conversation_id").distinct().count()
assert session_count == distinct_conv_ids, (
    f"Session count {session_count} != distinct conversation_id count "
    f"{distinct_conv_ids} — possible merge or split"
)
print(f"PASS: one session per conversation_id ({session_count})")


# COMMAND ----------

# MAGIC %md ## Publish as temp view for downstream notebooks

# COMMAND ----------

df_sessions.createOrReplaceTempView("sessions_abdullah_said")


# COMMAND ----------

# MAGIC %md ## Sample session record (complete ordered events)

# COMMAND ----------

sample = (
    spark.sql("""
        SELECT conversation_id, username, source_tool,
               started_at, ended_at, event_count, events
        FROM sessions_abdullah_said
        ORDER BY event_count DESC, conversation_id ASC
        LIMIT 1
    """)
    .first()
)

if sample is not None:
    print(f"\n=== SAMPLE SESSION (most events) ===")
    print(f"  conversation_id : {sample.conversation_id}")
    print(f"  username        : {sample.username}")
    print(f"  source_tool     : {sample.source_tool}")
    print(f"  started_at      : {sample.started_at}")
    print(f"  ended_at        : {sample.ended_at}")
    print(f"  event_count     : {sample.event_count}")
    print(f"\n  Ordered events ({len(sample.events)} total):")
    for ev in sample.events:
        print(f"    event_datetime={ev.event_datetime}  chat_step={ev.chat_step}")
        print(f"      query    : {(ev.query_text or '')[:120]!r}")
        print(f"      response : {(ev.response_text or '')[:120]!r}")
else:
    print("WARNING: no sessions — table may be empty")


# COMMAND ----------

# MAGIC %md ## Timeline overview

# COMMAND ----------

spark.sql("""
SELECT DATE_TRUNC('month', started_at) AS month,
       COUNT(*)            AS sessions,
       SUM(event_count)    AS events
FROM sessions_abdullah_said
GROUP BY 1
ORDER BY 1
""").show(20, truncate=False)



# COMMAND ----------

# Sentinel: must be the last thing that executes.
# The runner validates this output; SUCCESS without it means the notebook
# did not execute its full body.
import json as _j
dbutils.notebook.exit(_j.dumps({
    "sentinel":  "sessionize:OK",
    "sessions":  session_count,
}))
