# Databricks notebook source
# MAGIC %md
# MAGIC # Setup UC Objects — ai_fde_hackathon_catalog.automatic_user_context_profiles
# MAGIC
# MAGIC Creates all schema + tables as defined in SPEC §3.
# MAGIC All DDL uses CREATE TABLE IF NOT EXISTS — idempotent.

# COMMAND ----------

CATALOG = "ai_fde_hackathon_catalog"
SCHEMA  = "automatic_user_context_profiles"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------
# MAGIC %md ## §3.1 raw_conversations_abdullah_said

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.raw_conversations_abdullah_said (
  username         STRING    NOT NULL,
  query_text       STRING,
  response_text    STRING,
  chat_step        BIGINT    NOT NULL,
  conversation_id  STRING    NOT NULL,
  source_tool      STRING    NOT NULL,
  event_datetime   TIMESTAMP NOT NULL,
  ingested_at      TIMESTAMP NOT NULL,
  source_name      STRING    NOT NULL,
  row_hash         STRING    NOT NULL
)
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
print("raw_conversations_abdullah_said: OK")

# COMMAND ----------
# MAGIC %md ## §3.2 atomic_memories (SHARED CONTRACT)

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.atomic_memories (
  memory_id         STRING    NOT NULL,
  username          STRING    NOT NULL,
  memory_text       STRING    NOT NULL,
  memory_type       STRING    NOT NULL,
  domain            STRING    NOT NULL,
  source_tool       STRING    NOT NULL,
  conversation_id   STRING    NOT NULL,
  source_datetime   TIMESTAMP NOT NULL,
  evidence          STRING    NOT NULL,
  confidence        DOUBLE    NOT NULL,
  embedding         ARRAY<FLOAT>,
  embedding_model   STRING,
  extraction_model  STRING    NOT NULL,
  extraction_run_id STRING    NOT NULL,
  created_at        TIMESTAMP NOT NULL
)
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
print("atomic_memories: OK")

# COMMAND ----------
# MAGIC %md ## §3.3 eval_tasks

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.eval_tasks (
  task_id                  STRING        NOT NULL,
  goal_prompt              STRING        NOT NULL,
  repository               STRING        NOT NULL,
  starting_commit          STRING        NOT NULL,
  temporal_cutoff          TIMESTAMP     NOT NULL,
  heldout_conversation_ids ARRAY<STRING> NOT NULL,
  allowed_domains          ARRAY<STRING> NOT NULL,
  acceptance_command       STRING        NOT NULL,
  regression_command       STRING,
  required_files           ARRAY<STRING>,
  forbidden_files          ARRAY<STRING>,
  max_minutes              INT           NOT NULL,
  max_tool_calls           INT           NOT NULL,
  task_definition_hash     STRING        NOT NULL,
  created_at               TIMESTAMP     NOT NULL
)
USING DELTA
""")
print("eval_tasks: OK")

# COMMAND ----------
# MAGIC %md ## §3.4 memory_artifacts

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.memory_artifacts (
  artifact_id          STRING        NOT NULL,
  task_id              STRING        NOT NULL,
  arm                  STRING        NOT NULL,
  compiler_version     STRING        NOT NULL,
  config_json          STRING        NOT NULL,
  selected_memory_ids  ARRAY<STRING> NOT NULL,
  payload_sha256       STRING        NOT NULL,
  file_sha256          STRING        NOT NULL,
  sentinel             STRING        NOT NULL,
  memory_markdown      STRING        NOT NULL,
  token_count          INT           NOT NULL,
  volume_path          STRING        NOT NULL,
  frozen_at            TIMESTAMP     NOT NULL
)
USING DELTA
""")
print("memory_artifacts: OK")

# COMMAND ----------
# MAGIC %md ## §3.5 eval_runs

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.eval_runs (
  run_id                       STRING    NOT NULL,
  task_id                      STRING    NOT NULL,
  arm                          STRING    NOT NULL,
  repeat_number                INT       NOT NULL,
  artifact_id                  STRING    NOT NULL,
  file_sha256                  STRING    NOT NULL,
  sentinel_expected            STRING    NOT NULL,
  sentinel_observed            STRING,
  injection_verified           BOOLEAN   NOT NULL,
  agent_name                   STRING    NOT NULL,
  model_name                   STRING    NOT NULL,
  starting_commit              STRING    NOT NULL,
  final_commit                 STRING,
  success                      BOOLEAN   NOT NULL,
  acceptance_tests_passed      BOOLEAN   NOT NULL,
  regression_tests_passed      BOOLEAN,
  forbidden_files_unchanged    BOOLEAN   NOT NULL,
  elapsed_seconds              DOUBLE,
  total_tool_calls             INT,
  exploratory_reads_before_edit INT,
  failed_test_cycles           INT,
  input_tokens                 BIGINT,
  output_tokens                BIGINT,
  trace_path                   STRING,
  final_diff_path              STRING,
  failure_reason               STRING,
  started_at                   TIMESTAMP NOT NULL,
  completed_at                 TIMESTAMP
)
USING DELTA
""")
print("eval_runs: OK")

# COMMAND ----------
# MAGIC %md ## Verification

tables = spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
print(f"\nTables in {CATALOG}.{SCHEMA}:")
for t in tables:
    print(f"  {t.tableName}")
