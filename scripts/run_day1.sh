#!/usr/bin/env bash
# run_day1.sh — Execute Day-1 notebooks on fe-ai-sage once authenticated.
# Usage: bash scripts/run_day1.sh [fe-ai-sage]
set -euo pipefail

PROFILE="${1:-fe-ai-sage}"
WORKSPACE_PATH="/Users/abdullah.said@databricks.com/hackathon_auto_profiles"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CSV_DBFS="/dbfs/tmp/raw_abdullah_tab.csv"

echo "=== Day-1 Databricks Execution ==="
echo "Profile  : $PROFILE"
echo "Workspace: $WORKSPACE_PATH"
echo ""

# 0. Verify auth
echo "[0] Verifying auth..."
databricks current-user me --profile "$PROFILE" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('  Authenticated as:', d.get('userName'))
"

# 1. Pick a warehouse
echo "[1] Finding warehouse..."
WH_ID=$(databricks warehouses list --profile "$PROFILE" --output json | python3 -c "
import json,sys
ws=json.load(sys.stdin)
running=[w for w in ws if w.get('state')=='RUNNING']
all_ws=running or ws
print(sorted(all_ws, key=lambda w: w.get('num_active_sessions',0), reverse=True)[0]['id'])
")
echo "  Warehouse: $WH_ID"

# 2. Upload CSV to DBFS (fallback input for notebook 00)
echo "[2] Uploading CSV to DBFS..."
databricks fs cp "$REPO_ROOT/data/raw_abdullah_tab.csv" "dbfs:/tmp/raw_abdullah_tab.csv" \
  --profile "$PROFILE" --overwrite
echo "  CSV uploaded to dbfs:/tmp/raw_abdullah_tab.csv"

# 3. Import notebooks to workspace
echo "[3] Importing notebooks..."
for nb in _setup_uc_objects 00_ingest_abdullah_tab 01_sessionize; do
    databricks workspace import "$WORKSPACE_PATH/$nb" \
      --file "$REPO_ROOT/notebooks/${nb}.py" \
      --language PYTHON \
      --format SOURCE \
      --overwrite \
      --profile "$PROFILE"
    echo "  Imported: $nb"
done

# 4. Run notebooks in order via one-time jobs
run_notebook() {
    local NB_PATH="$1"
    local JOB_NAME="$2"
    local CLUSTER_KEY="$3"

    echo "[run] $NB_PATH"

    # Create one-time run
    RUN_ID=$(databricks runs submit \
      --json "{
        \"run_name\": \"$JOB_NAME\",
        \"new_cluster\": {
          \"spark_version\": \"15.4.x-scala2.12\",
          \"node_type_id\": \"i3.xlarge\",
          \"num_workers\": 1,
          \"spark_conf\": {\"spark.databricks.cluster.profile\": \"singleNode\"},
          \"single_user_name\": \"$(databricks current-user me --profile $PROFILE | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"userName\"])')\"
        },
        \"notebook_task\": {\"notebook_path\": \"$NB_PATH\"}
      }" \
      --profile "$PROFILE" | python3 -c "import json,sys; print(json.load(sys.stdin)['run_id'])")

    echo "  run_id=$RUN_ID — waiting..."
    databricks runs wait --run-id "$RUN_ID" --profile "$PROFILE"

    STATE=$(databricks runs get --run-id "$RUN_ID" --profile "$PROFILE" | \
      python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state']['result_state'])")
    echo "  State: $STATE"
    if [ "$STATE" != "SUCCESS" ]; then
        echo "  FAILED — check run: databricks runs get --run-id $RUN_ID --profile $PROFILE"
        exit 1
    fi
}

# Use SQL warehouse for setup + ingest to avoid cluster spin-up
echo ""
echo "[4] Executing notebooks..."

# _setup_uc_objects via SQL statements
echo "  Creating UC objects via SQL..."
for SQL in \
    "CREATE SCHEMA IF NOT EXISTS ai_fde_hackathon_catalog.automatic_user_context_profiles" \
    "CREATE TABLE IF NOT EXISTS ai_fde_hackathon_catalog.automatic_user_context_profiles.raw_conversations_abdullah_said (username STRING NOT NULL, query_text STRING, response_text STRING, chat_step BIGINT NOT NULL, conversation_id STRING NOT NULL, source_tool STRING NOT NULL, event_datetime TIMESTAMP NOT NULL, ingested_at TIMESTAMP NOT NULL, source_name STRING NOT NULL, row_hash STRING NOT NULL) USING DELTA TBLPROPERTIES ('delta.enableChangeDataFeed'='true')" \
    "CREATE TABLE IF NOT EXISTS ai_fde_hackathon_catalog.automatic_user_context_profiles.atomic_memories (memory_id STRING NOT NULL, username STRING NOT NULL, memory_text STRING NOT NULL, memory_type STRING NOT NULL, domain STRING NOT NULL, source_tool STRING NOT NULL, conversation_id STRING NOT NULL, source_datetime TIMESTAMP NOT NULL, evidence STRING NOT NULL, confidence DOUBLE NOT NULL, embedding ARRAY<FLOAT>, embedding_model STRING, extraction_model STRING NOT NULL, extraction_run_id STRING NOT NULL, created_at TIMESTAMP NOT NULL) USING DELTA TBLPROPERTIES ('delta.enableChangeDataFeed'='true')" \
    "CREATE TABLE IF NOT EXISTS ai_fde_hackathon_catalog.automatic_user_context_profiles.eval_tasks (task_id STRING NOT NULL, goal_prompt STRING NOT NULL, repository STRING NOT NULL, starting_commit STRING NOT NULL, temporal_cutoff TIMESTAMP NOT NULL, heldout_conversation_ids ARRAY<STRING> NOT NULL, allowed_domains ARRAY<STRING> NOT NULL, acceptance_command STRING NOT NULL, regression_command STRING, required_files ARRAY<STRING>, forbidden_files ARRAY<STRING>, max_minutes INT NOT NULL, max_tool_calls INT NOT NULL, task_definition_hash STRING NOT NULL, created_at TIMESTAMP NOT NULL) USING DELTA" \
    "CREATE TABLE IF NOT EXISTS ai_fde_hackathon_catalog.automatic_user_context_profiles.memory_artifacts (artifact_id STRING NOT NULL, task_id STRING NOT NULL, arm STRING NOT NULL, compiler_version STRING NOT NULL, config_json STRING NOT NULL, selected_memory_ids ARRAY<STRING> NOT NULL, payload_sha256 STRING NOT NULL, file_sha256 STRING NOT NULL, sentinel STRING NOT NULL, memory_markdown STRING NOT NULL, token_count INT NOT NULL, volume_path STRING NOT NULL, frozen_at TIMESTAMP NOT NULL) USING DELTA" \
    "CREATE TABLE IF NOT EXISTS ai_fde_hackathon_catalog.automatic_user_context_profiles.eval_runs (run_id STRING NOT NULL, task_id STRING NOT NULL, arm STRING NOT NULL, repeat_number INT NOT NULL, artifact_id STRING NOT NULL, file_sha256 STRING NOT NULL, sentinel_expected STRING NOT NULL, sentinel_observed STRING, injection_verified BOOLEAN NOT NULL, agent_name STRING NOT NULL, model_name STRING NOT NULL, starting_commit STRING NOT NULL, final_commit STRING, success BOOLEAN NOT NULL, acceptance_tests_passed BOOLEAN NOT NULL, regression_tests_passed BOOLEAN, forbidden_files_unchanged BOOLEAN NOT NULL, elapsed_seconds DOUBLE, total_tool_calls INT, exploratory_reads_before_edit INT, failed_test_cycles INT, input_tokens BIGINT, output_tokens BIGINT, trace_path STRING, final_diff_path STRING, failure_reason STRING, started_at TIMESTAMP NOT NULL, completed_at TIMESTAMP) USING DELTA" \
; do
    OUT=$(databricks api post /api/2.0/sql/statements/ \
      --json "{\"statement\": $(python3 -c "import json,sys; print(json.dumps('$SQL'))"), \"warehouse_id\": \"$WH_ID\", \"format\":\"JSON_ARRAY\", \"wait_timeout\":\"60s\"}" \
      --profile "$PROFILE" 2>&1)
    STATE=$(echo "$OUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status',{}).get('state','?'))" 2>/dev/null || echo "PARSE_ERROR")
    echo "    $STATE — ${SQL:0:80}..."
done

echo ""
echo "=== Day-1 setup complete. Run notebooks 00 + 01 via Databricks UI or workspace runs. ==="
echo "Notebook paths:"
echo "  $WORKSPACE_PATH/_setup_uc_objects"
echo "  $WORKSPACE_PATH/00_ingest_abdullah_tab"
echo "  $WORKSPACE_PATH/01_sessionize"
