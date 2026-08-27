# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup UC Objects — ai_fde_hackathon_catalog.automatic_user_context_profiles
# MAGIC
# MAGIC Creates all schema + tables as specified in SPEC §3.
# MAGIC All DDL uses CREATE TABLE IF NOT EXISTS (idempotent).
# MAGIC Each table is verified against the SPEC schema after creation; the notebook
# MAGIC fails loudly on any column / type / nullability drift.


# COMMAND ----------


from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, DoubleType, BooleanType, IntegerType,
    TimestampType, ArrayType, FloatType,
)

CATALOG = "ai_fde_hackathon_catalog"
SCHEMA  = "automatic_user_context_profiles"

# ---------------------------------------------------------------------------
# SPEC §3 canonical schema — the ground truth for drift detection.
# (name, DataType, nullable)
# ---------------------------------------------------------------------------
EXPECTED_SCHEMA: dict[str, list[tuple]] = {
    # §3.1
    "raw_conversations_abdullah_said": [
        ("username",        StringType(),    False),
        ("query_text",      StringType(),    True),
        ("response_text",   StringType(),    True),
        ("chat_step",       LongType(),      False),
        ("conversation_id", StringType(),    False),
        ("source_tool",     StringType(),    False),
        ("event_datetime",  TimestampType(), False),
        ("ingested_at",     TimestampType(), False),
        ("source_name",     StringType(),    False),
        ("row_hash",        StringType(),    False),
    ],
    # §3.2 SHARED CONTRACT
    "atomic_memories": [
        ("memory_id",         StringType(),            False),
        ("username",          StringType(),            False),
        ("memory_text",       StringType(),            False),
        ("memory_type",       StringType(),            False),
        ("domain",            StringType(),            False),
        ("source_tool",       StringType(),            False),
        ("conversation_id",   StringType(),            False),
        ("source_datetime",   TimestampType(),         False),
        ("evidence",          StringType(),            False),
        ("confidence",        DoubleType(),            False),
        ("embedding",         ArrayType(FloatType()),  True),
        ("embedding_model",   StringType(),            True),
        ("extraction_model",  StringType(),            False),
        ("extraction_run_id", StringType(),            False),
        ("created_at",        TimestampType(),         False),
    ],
    # §3.3
    "eval_tasks": [
        ("task_id",                  StringType(),             False),
        ("goal_prompt",              StringType(),             False),
        ("repository",               StringType(),             False),
        ("starting_commit",          StringType(),             False),
        ("temporal_cutoff",          TimestampType(),          False),
        ("heldout_conversation_ids", ArrayType(StringType()), False),
        ("allowed_domains",          ArrayType(StringType()), False),
        ("acceptance_command",       StringType(),             False),
        ("regression_command",       StringType(),             True),
        ("required_files",           ArrayType(StringType()), True),
        ("forbidden_files",          ArrayType(StringType()), True),
        ("max_minutes",              IntegerType(),            False),
        ("max_tool_calls",           IntegerType(),            False),
        ("task_definition_hash",     StringType(),             False),
        ("created_at",               TimestampType(),          False),
    ],
    # §3.4
    "memory_artifacts": [
        ("artifact_id",         StringType(),             False),
        ("task_id",             StringType(),             False),
        ("arm",                 StringType(),             False),
        ("compiler_version",    StringType(),             False),
        ("config_json",         StringType(),             False),
        ("selected_memory_ids", ArrayType(StringType()), False),
        ("payload_sha256",      StringType(),             False),
        ("file_sha256",         StringType(),             False),
        ("sentinel",            StringType(),             False),
        ("memory_markdown",     StringType(),             False),
        ("token_count",         IntegerType(),            False),
        ("volume_path",         StringType(),             False),
        ("frozen_at",           TimestampType(),          False),
    ],
    # §3.5
    "eval_runs": [
        ("run_id",                        StringType(),    False),
        ("task_id",                       StringType(),    False),
        ("arm",                           StringType(),    False),
        ("repeat_number",                 IntegerType(),   False),
        ("artifact_id",                   StringType(),    False),
        ("file_sha256",                   StringType(),    False),
        ("sentinel_expected",             StringType(),    False),
        ("sentinel_observed",             StringType(),    True),
        ("injection_verified",            BooleanType(),   False),
        ("agent_name",                    StringType(),    False),
        ("model_name",                    StringType(),    False),
        ("starting_commit",               StringType(),    False),
        ("final_commit",                  StringType(),    True),
        ("success",                       BooleanType(),   False),
        ("acceptance_tests_passed",       BooleanType(),   False),
        ("regression_tests_passed",       BooleanType(),   True),
        ("forbidden_files_unchanged",     BooleanType(),   False),
        ("elapsed_seconds",               DoubleType(),    True),
        ("total_tool_calls",              IntegerType(),   True),
        ("exploratory_reads_before_edit", IntegerType(),   True),
        ("failed_test_cycles",            IntegerType(),   True),
        ("input_tokens",                  LongType(),      True),
        ("output_tokens",                 LongType(),      True),
        ("trace_path",                    StringType(),    True),
        ("final_diff_path",               StringType(),    True),
        ("failure_reason",                StringType(),    True),
        ("started_at",                    TimestampType(), False),
        ("completed_at",                  TimestampType(), True),
    ],
}


def verify_schema(table_short: str, full_name: str) -> None:
    """Fail loudly if the live table schema diverges from SPEC."""
    expected = EXPECTED_SCHEMA[table_short]
    actual_fields = {f.name: f for f in spark.table(full_name).schema.fields}
    exp_fields    = {name: (dt, nullable) for name, dt, nullable in expected}

    errors = []

    missing = set(exp_fields) - set(actual_fields)
    extra   = set(actual_fields) - set(exp_fields)
    if missing:
        errors.append(f"  MISSING columns: {sorted(missing)}")
    if extra:
        errors.append(f"  EXTRA columns:   {sorted(extra)}")

    for col in sorted(set(exp_fields) & set(actual_fields)):
        exp_dt, exp_null = exp_fields[col]
        act_f = actual_fields[col]
        if exp_dt.simpleString() != act_f.dataType.simpleString():
            errors.append(
                f"  TYPE DRIFT [{col}]: "
                f"expected={exp_dt.simpleString()} actual={act_f.dataType.simpleString()}"
            )
        if exp_null != act_f.nullable:
            errors.append(
                f"  NULLABLE DRIFT [{col}]: "
                f"expected={exp_null} actual={act_f.nullable}"
            )

    if errors:
        raise ValueError(
            f"Schema drift detected in {full_name}:\n" + "\n".join(errors)
        )
    print(f"  schema OK — {len(actual_fields)} columns match SPEC")



# COMMAND ----------

# MAGIC %md ## Create schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema: {CATALOG}.{SCHEMA}")


# COMMAND ----------

# MAGIC %md ## §3.1 raw_conversations_abdullah_said

# COMMAND ----------

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
verify_schema("raw_conversations_abdullah_said",
              f"{CATALOG}.{SCHEMA}.raw_conversations_abdullah_said")
print("raw_conversations_abdullah_said: OK")


# COMMAND ----------

# MAGIC %md ## §3.2 atomic_memories (SHARED CONTRACT — exact schema required)

# COMMAND ----------

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
verify_schema("atomic_memories", f"{CATALOG}.{SCHEMA}.atomic_memories")
print("atomic_memories: OK")


# COMMAND ----------

# MAGIC %md ## §3.3 eval_tasks

# COMMAND ----------

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
verify_schema("eval_tasks", f"{CATALOG}.{SCHEMA}.eval_tasks")
print("eval_tasks: OK")


# COMMAND ----------

# MAGIC %md ## §3.4 memory_artifacts

# COMMAND ----------

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
verify_schema("memory_artifacts", f"{CATALOG}.{SCHEMA}.memory_artifacts")
print("memory_artifacts: OK")


# COMMAND ----------

# MAGIC %md ## §3.5 eval_runs

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.eval_runs (
  run_id                        STRING    NOT NULL,
  task_id                       STRING    NOT NULL,
  arm                           STRING    NOT NULL,
  repeat_number                 INT       NOT NULL,
  artifact_id                   STRING    NOT NULL,
  file_sha256                   STRING    NOT NULL,
  sentinel_expected             STRING    NOT NULL,
  sentinel_observed             STRING,
  injection_verified            BOOLEAN   NOT NULL,
  agent_name                    STRING    NOT NULL,
  model_name                    STRING    NOT NULL,
  starting_commit               STRING    NOT NULL,
  final_commit                  STRING,
  success                       BOOLEAN   NOT NULL,
  acceptance_tests_passed       BOOLEAN   NOT NULL,
  regression_tests_passed       BOOLEAN,
  forbidden_files_unchanged     BOOLEAN   NOT NULL,
  elapsed_seconds               DOUBLE,
  total_tool_calls              INT,
  exploratory_reads_before_edit INT,
  failed_test_cycles            INT,
  input_tokens                  BIGINT,
  output_tokens                 BIGINT,
  trace_path                    STRING,
  final_diff_path               STRING,
  failure_reason                STRING,
  started_at                    TIMESTAMP NOT NULL,
  completed_at                  TIMESTAMP
)
USING DELTA
""")
verify_schema("eval_runs", f"{CATALOG}.{SCHEMA}.eval_runs")
print("eval_runs: OK")


# COMMAND ----------

# MAGIC %md ## Final verification — all 5 tables present with correct schemas

# COMMAND ----------

REQUIRED_TABLES = set(EXPECTED_SCHEMA.keys())
actual_tables = {
    r.tableName
    for r in spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
}
missing = REQUIRED_TABLES - actual_tables
assert not missing, f"Missing tables: {missing}"

print(f"\nAll {len(REQUIRED_TABLES)} required tables present and schema-verified:")
for tbl in sorted(REQUIRED_TABLES):
    print(f"  {CATALOG}.{SCHEMA}.{tbl}")



# COMMAND ----------

# Sentinel: must be the last thing that executes.
# The runner validates this output; SUCCESS without it means the notebook
# did not execute its full body.
import json as _j
dbutils.notebook.exit(_j.dumps({"sentinel": "setup:OK"}))
