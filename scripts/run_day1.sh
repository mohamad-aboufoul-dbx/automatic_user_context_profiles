#!/usr/bin/env bash
# run_day1.sh — Execute Day-1 notebooks on the target Databricks workspace.
# Usage: bash scripts/run_day1.sh [profile]
# Default profile: fe-ai-sage
#
# Runs in order: _setup_uc_objects → 00_ingest_abdullah_tab → 01_sessionize
# Each notebook runs on its own cluster job and must succeed before the next starts.
# Any non-SUCCESS result exits 1 immediately.
# Safe to re-run (MERGE is idempotent, CREATE TABLE IF NOT EXISTS is idempotent).
set -euo pipefail

PROFILE="${1:-fe-ai-sage}"
WORKSPACE_PATH="/Users/abdullah.said@databricks.com/hackathon_auto_profiles"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POLL_INTERVAL=20   # seconds between lifecycle state polls
MAX_WAIT=1800      # abort if a run has not completed in 30 minutes
SPARK_VERSION="15.4.x-scala2.12"
NODE_TYPE="i3.xlarge"
NUM_WORKERS=1      # standard single-worker cluster (no conflicting singleNode profile)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"; }

require_cmd databricks
require_cmd python3

# ---------------------------------------------------------------------------
# 0. Verify auth
# ---------------------------------------------------------------------------
echo "=== Day-1 Databricks Execution ==="
echo "Profile : $PROFILE"
echo ""
echo "[0] Verifying authentication..."
ME=$(databricks current-user me --profile "$PROFILE" \
     | python3 -c "import json,sys; print(json.load(sys.stdin)['userName'])" 2>&1) \
  || die "Auth check failed. Run: databricks auth login --host https://fe-ai-sage.cloud.databricks.com --profile $PROFILE"
echo "    Authenticated as: $ME"

# ---------------------------------------------------------------------------
# 1. Upload CSV fallback to DBFS
# ---------------------------------------------------------------------------
echo "[1] Uploading CSV fallback to DBFS..."
CSV_SRC="$REPO_ROOT/data/raw_abdullah_tab.csv"
[ -f "$CSV_SRC" ] || die "CSV not found: $CSV_SRC"
databricks fs cp "$CSV_SRC" "dbfs:/tmp/raw_abdullah_tab.csv" \
  --profile "$PROFILE" --overwrite
echo "    Uploaded: dbfs:/tmp/raw_abdullah_tab.csv"

# ---------------------------------------------------------------------------
# 2. Import notebooks to workspace
# ---------------------------------------------------------------------------
echo "[2] Importing notebooks to workspace..."
for nb in _setup_uc_objects 00_ingest_abdullah_tab 01_sessionize; do
    databricks workspace import "$WORKSPACE_PATH/$nb" \
      --file "$REPO_ROOT/notebooks/${nb}.py" \
      --language PYTHON \
      --format SOURCE \
      --overwrite \
      --profile "$PROFILE"
    echo "    Imported: $WORKSPACE_PATH/$nb"
done

# ---------------------------------------------------------------------------
# helpers: submit and wait
# ---------------------------------------------------------------------------

# submit_notebook_run <notebook_path> <run_name> <profile>
# Echoes the run_id to stdout.
submit_notebook_run() {
    local NB_PATH="$1"
    local RUN_NAME="$2"
    local PROFILE="$3"

    local PAYLOAD
    PAYLOAD=$(python3 - <<PY
import json
payload = {
    "run_name": "$RUN_NAME",
    "new_cluster": {
        "spark_version": "$SPARK_VERSION",
        "node_type_id":  "$NODE_TYPE",
        "num_workers":    $NUM_WORKERS,
    },
    "notebook_task": {
        "notebook_path": "$NB_PATH",
        "source":        "WORKSPACE",
    },
}
print(json.dumps(payload))
PY
)
    databricks runs submit --json "$PAYLOAD" --profile "$PROFILE" \
      | python3 -c "import json,sys; print(json.load(sys.stdin)['run_id'])"
}

# wait_for_run <run_id> <profile>
# Polls until the run reaches a terminal lifecycle state.
# Exits 1 if result_state is not SUCCESS.
wait_for_run() {
    local RUN_ID="$1"
    local PROFILE="$2"
    local elapsed=0

    while [ $elapsed -lt $MAX_WAIT ]; do
        local LC_STATE
        LC_STATE=$(databricks runs get --run-id "$RUN_ID" --profile "$PROFILE" \
            | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(d['state']['life_cycle_state'])
")
        case "$LC_STATE" in
            TERMINATED|INTERNAL_ERROR|SKIPPED)
                break ;;
            *)
                printf "    ... %s (%ds elapsed)\r" "$LC_STATE" "$elapsed"
                sleep $POLL_INTERVAL
                elapsed=$((elapsed + POLL_INTERVAL))
                ;;
        esac
    done
    echo ""  # clear the carriage-return line

    if [ $elapsed -ge $MAX_WAIT ]; then
        die "Timed out waiting for run_id=$RUN_ID after ${MAX_WAIT}s"
    fi

    local RESULT_STATE
    RESULT_STATE=$(databricks runs get --run-id "$RUN_ID" --profile "$PROFILE" \
        | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(d['state'].get('result_state','UNKNOWN'))
")
    if [ "$RESULT_STATE" != "SUCCESS" ]; then
        echo "    RESULT: $RESULT_STATE"
        echo "    Inspect run: databricks runs get --run-id $RUN_ID --profile $PROFILE"
        die "Notebook run failed (run_id=$RUN_ID result=$RESULT_STATE)"
    fi
    echo "    RESULT: $RESULT_STATE"
}

# ---------------------------------------------------------------------------
# 3. Run notebooks in order
# ---------------------------------------------------------------------------
echo "[3] Running notebooks in order..."
echo ""

for ENTRY in \
    "_setup_uc_objects:day1-setup" \
    "00_ingest_abdullah_tab:day1-ingest" \
    "01_sessionize:day1-sessionize" \
; do
    NB_SHORT="${ENTRY%%:*}"
    RUN_NAME="${ENTRY##*:}"
    NB_FULL="$WORKSPACE_PATH/$NB_SHORT"

    echo "  [run] $NB_FULL"
    RUN_ID=$(submit_notebook_run "$NB_FULL" "$RUN_NAME" "$PROFILE")
    echo "    run_id=$RUN_ID — waiting for completion..."
    wait_for_run "$RUN_ID" "$PROFILE"
done

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== All Day-1 notebooks completed successfully ==="
echo "    _setup_uc_objects   : UC schema + all 5 tables created and verified"
echo "    00_ingest_abdullah_tab : data loaded into raw_conversations_abdullah_said"
echo "    01_sessionize          : sessions view populated"
