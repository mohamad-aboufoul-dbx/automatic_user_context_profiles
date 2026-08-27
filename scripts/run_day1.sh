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
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POLL_INTERVAL=20   # seconds between lifecycle state polls
MAX_WAIT=1800      # abort if a run has not completed in 30 minutes
# WORKSPACE_PATH is derived from the authenticated user after step [0].

# Unity Catalog staging target (public DBFS root is disabled on this workspace).
CATALOG="ai_fde_hackathon_catalog"
SCHEMA="automatic_user_context_profiles"
VOLUME="raw"
CSV_VOLUME_DEST="dbfs:/Volumes/${CATALOG}/${SCHEMA}/${VOLUME}/raw_abdullah_tab.csv"

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
WORKSPACE_PATH="/Users/$ME/hackathon_auto_profiles"

# ---------------------------------------------------------------------------
# 1. Ensure UC schema + volume exist, then upload CSV fallback to the volume
# ---------------------------------------------------------------------------
# Public DBFS root is disabled on this workspace, so the CSV is staged in a
# Unity Catalog managed volume. Schema and volume must exist BEFORE the upload.
echo "[1] Ensuring UC schema + volume, then uploading CSV..."
CSV_SRC="$REPO_ROOT/data/raw_abdullah_tab.csv"
[ -f "$CSV_SRC" ] || die "CSV not found: $CSV_SRC"

# Schema (UC control-plane calls are synchronous; create only if missing).
if databricks schemas get "${CATALOG}.${SCHEMA}" --profile "$PROFILE" >/dev/null 2>&1; then
    echo "    Schema exists: ${CATALOG}.${SCHEMA}"
else
    databricks schemas create "$SCHEMA" "$CATALOG" --profile "$PROFILE" >/dev/null \
      || die "Failed to create schema ${CATALOG}.${SCHEMA}"
    echo "    Schema created: ${CATALOG}.${SCHEMA}"
fi

# Managed volume (create only if missing).
if databricks volumes read "${CATALOG}.${SCHEMA}.${VOLUME}" --profile "$PROFILE" >/dev/null 2>&1; then
    echo "    Volume exists: ${CATALOG}.${SCHEMA}.${VOLUME}"
else
    databricks volumes create "$CATALOG" "$SCHEMA" "$VOLUME" MANAGED --profile "$PROFILE" >/dev/null \
      || die "Failed to create volume ${CATALOG}.${SCHEMA}.${VOLUME}"
    echo "    Volume created: ${CATALOG}.${SCHEMA}.${VOLUME}"
fi

# Upload CSV to the volume (fs cp routes /Volumes paths through the Files API).
databricks fs cp "$CSV_SRC" "$CSV_VOLUME_DEST" \
  --profile "$PROFILE" --overwrite \
  || die "Failed to upload CSV to $CSV_VOLUME_DEST"
echo "    Uploaded: $CSV_VOLUME_DEST"

# ---------------------------------------------------------------------------
# 2. Import notebooks to workspace
# ---------------------------------------------------------------------------
echo "[2] Importing notebooks to workspace..."
# Ensure the parent workspace folder exists (mkdirs creates parents and is a
# no-op if the folder already exists). Import fails if the parent is missing.
databricks workspace mkdirs "$WORKSPACE_PATH" --profile "$PROFILE" \
  || die "Failed to create workspace directory $WORKSPACE_PATH"
echo "    Workspace dir ready: $WORKSPACE_PATH"
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
# Submits a serverless one-time run and echoes the run_id to stdout.
# Uses the current CLI: `databricks jobs submit` with tasks-array payload.
submit_notebook_run() {
    local NB_PATH="$1"
    local RUN_NAME="$2"
    local PROFILE="$3"

    local PAYLOAD
    PAYLOAD=$(python3 - <<PY
import json
payload = {
    "run_name": "$RUN_NAME",
    "tasks": [
        {
            "task_key": "main",
            "notebook_task": {
                "notebook_path": "$NB_PATH",
                "source":        "WORKSPACE",
            },
        }
    ],
}
print(json.dumps(payload))
PY
)
    databricks jobs submit --no-wait --json "$PAYLOAD" -o json --profile "$PROFILE" \
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
        LC_STATE=$(databricks jobs get-run "$RUN_ID" -o json --profile "$PROFILE" \
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
    RESULT_STATE=$(databricks jobs get-run "$RUN_ID" -o json --profile "$PROFILE" \
        | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(d['state'].get('result_state','UNKNOWN'))
")
    if [ "$RESULT_STATE" != "SUCCESS" ]; then
        echo "    RESULT: $RESULT_STATE"
        echo "    Inspect run: databricks jobs get-run $RUN_ID --profile $PROFILE"
        die "Notebook run failed (run_id=$RUN_ID result=$RESULT_STATE)"
    fi
    echo "    RESULT: $RESULT_STATE"
}

# validate_sentinel <parent_run_id> <expected_sentinel> <profile>
# Validates that the notebook called dbutils.notebook.exit with the expected
# sentinel JSON. A run that only executed its first cell (imports) would never
# reach the exit call, so a missing/wrong sentinel proves incomplete execution.
validate_sentinel() {
    local RUN_ID="$1"
    local EXPECTED="$2"
    local PROFILE="$3"

    # Get the task-level run_id from the parent run
    local TASK_RUN_ID
    TASK_RUN_ID=$(databricks jobs get-run "$RUN_ID" -o json --profile "$PROFILE" \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['tasks'][0]['run_id'])")

    # Pipe the CLI JSON straight into a standalone parser script. Because there
    # is no heredoc, stdin carries the piped JSON (the previous `python3 - <<PY`
    # form let the heredoc shadow stdin, so json.load(sys.stdin) saw nothing and
    # raised JSONDecodeError). The expected sentinel is passed as argv, not
    # interpolated into Python source.
    databricks jobs get-run-output "$TASK_RUN_ID" -o json --profile "$PROFILE" \
      | python3 "$REPO_ROOT/scripts/_check_sentinel.py" "$EXPECTED" \
      || die "Sentinel validation FAILED for run_id=$RUN_ID (task=$TASK_RUN_ID)"
}

# ---------------------------------------------------------------------------
# 3. Run notebooks in order
# ---------------------------------------------------------------------------
echo "[3] Running notebooks in order..."
echo ""

for ENTRY in \
    "_setup_uc_objects:day1-setup:setup:OK" \
    "00_ingest_abdullah_tab:day1-ingest:ingest:OK" \
    "01_sessionize:day1-sessionize:sessionize:OK" \
; do
    NB_SHORT="${ENTRY%%:*}"
    REST="${ENTRY#*:}"
    RUN_NAME="${REST%%:*}"
    SENTINEL="${REST#*:}"
    NB_FULL="$WORKSPACE_PATH/$NB_SHORT"

    echo "  [run] $NB_FULL"
    RUN_ID=$(submit_notebook_run "$NB_FULL" "$RUN_NAME" "$PROFILE")
    echo "    run_id=$RUN_ID — waiting for completion..."
    wait_for_run "$RUN_ID" "$PROFILE"
    validate_sentinel "$RUN_ID" "$SENTINEL" "$PROFILE"
done

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== All Day-1 notebooks completed successfully ==="
echo "    _setup_uc_objects   : UC schema + all 5 tables created and verified"
echo "    00_ingest_abdullah_tab : data loaded into raw_conversations_abdullah_said"
echo "    01_sessionize          : sessions view populated"
