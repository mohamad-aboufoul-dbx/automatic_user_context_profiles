#!/usr/bin/env bash
# run_compile.sh — Run notebook 04 (compile_memory_md) on Databricks.
# Usage: bash scripts/run_compile.sh [dry_run|full] [profile]
#   arg1  dry_run|full   controls the submitted `dry_run` notebook param
#                        (default: dry_run — SAFE: full compute + assertions, writes nothing)
#   arg2  profile        Databricks CLI profile (default: hackathon)
#
# Stages src/extract/*.py + src/compiler/*.py + config JSONs + eval task JSONs
# to the UC Volume, pre-creates the artifact dirs for all 12 (task, arm)
# combinations, imports the 04 notebook, submits ONE serverless run passing
# `dry_run`, waits, validates the sentinel, and prints the run output.
#
# This runner does NOT run earlier notebooks and does NOT loop. It is the
# single, deliberate entrypoint for artifact compilation, gated behind the
# human-approval checkpoint. Safe to re-run (MERGE on artifact_id is
# idempotent; Volume writes use overwrite=True).
set -euo pipefail

MODE="${1:-dry_run}"
PROFILE="${2:-hackathon}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POLL_INTERVAL=20    # seconds between lifecycle state polls
MAX_WAIT=3600       # abort if a run has not completed in 60 minutes

# Map MODE -> the dry_run param value + the expected sentinel.
case "$MODE" in
    dry_run)
        DRY_RUN_PARAM="true"
        EXPECTED_SENTINEL="compile_dryrun:OK"
        ;;
    full)
        DRY_RUN_PARAM="false"
        EXPECTED_SENTINEL="compile:OK"
        ;;
    *)
        echo "ERROR: first arg must be 'dry_run' or 'full' (got '$MODE')" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Unity Catalog staging targets (public DBFS root is disabled on this workspace).
# These paths MUST match the constants in notebooks/04_compile_memory_md.py:
#   EXTRACT_SRC_DIR  = /Volumes/<cat>/<schema>/raw/extract_src   (parent of `extract` pkg)
#   COMPILER_SRC_DIR = /Volumes/<cat>/<schema>/raw/compiler_src  (parent of `compiler` pkg)
#   CONFIG_DIR       = /Volumes/<cat>/<schema>/raw/config
#   EVAL_TASKS_DIR   = /Volumes/<cat>/<schema>/raw/eval_tasks
#   ARTIFACTS_BASE_DIR = /Volumes/<cat>/<schema>/raw/artifacts
# ---------------------------------------------------------------------------
CATALOG="ai_fde_hackathon_catalog"
SCHEMA="automatic_user_context_profiles"
VOLUME="raw"
VOLUME_ROOT="dbfs:/Volumes/${CATALOG}/${SCHEMA}/${VOLUME}"
EXTRACT_SRC_DEST="${VOLUME_ROOT}/extract_src/extract"    # package dir on the volume
COMPILER_SRC_DEST="${VOLUME_ROOT}/compiler_src/compiler" # package dir on the volume
CONFIG_DEST="${VOLUME_ROOT}/config"
EVAL_TASKS_DEST="${VOLUME_ROOT}/eval_tasks"
ARTIFACTS_DEST="${VOLUME_ROOT}/artifacts"

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
echo "=== Compile (notebook 04) — Databricks Execution ==="
echo "Mode    : $MODE  (dry_run param = $DRY_RUN_PARAM)"
echo "Profile : $PROFILE"
echo ""
echo "[0] Verifying authentication..."
ME=$(databricks current-user me --profile "$PROFILE" \
     | python3 -c "import json,sys; print(json.load(sys.stdin)['userName'])" 2>&1) \
  || die "Auth check failed. Run: databricks auth login --profile $PROFILE"
echo "    Authenticated as: $ME"
WORKSPACE_PATH="/Users/$ME/hackathon_auto_profiles"

# ---------------------------------------------------------------------------
# 1. Stage source packages, config, and eval task files to the UC Volume
# ---------------------------------------------------------------------------
echo "[1] Staging source packages + config + eval tasks to the UC Volume..."

# `databricks fs cp` does NOT auto-create nested Volume subdirectories, so make
# the target dirs first. `fs mkdir` creates every dir along the path and is a
# no-op on an existing dir (idempotent / re-runnable).
for d in "$EXTRACT_SRC_DEST" "$COMPILER_SRC_DEST" "$CONFIG_DEST" "$EVAL_TASKS_DEST"; do
    databricks fs mkdir "$d" --profile "$PROFILE" \
      || die "Failed to create Volume directory $d"
    echo "    Ensured dir: $d"
done

# extract package: stage EVERY src/extract/*.py so `import extract.*` resolves.
for f in __init__.py prefilter.py schema.py ids.py; do
    SRC="$REPO_ROOT/src/extract/$f"
    [ -f "$SRC" ] || die "Extract module not found: $SRC"
    databricks fs cp "$SRC" "$EXTRACT_SRC_DEST/$f" \
      --profile "$PROFILE" --overwrite \
      || die "Failed to stage $SRC -> $EXTRACT_SRC_DEST/$f"
    echo "    Staged: $EXTRACT_SRC_DEST/$f"
done

# compiler package: stage EVERY src/compiler/*.py so `import compiler.*` resolves.
for f in __init__.py arms.py render.py score.py artifact.py; do
    SRC="$REPO_ROOT/src/compiler/$f"
    [ -f "$SRC" ] || die "Compiler module not found: $SRC"
    databricks fs cp "$SRC" "$COMPILER_SRC_DEST/$f" \
      --profile "$PROFILE" --overwrite \
      || die "Failed to stage $SRC -> $COMPILER_SRC_DEST/$f"
    echo "    Staged: $COMPILER_SRC_DEST/$f"
done

# Config JSONs (read from the Volume by the notebook).
for f in compile_config.json heldout_exclusions.json; do
    SRC="$REPO_ROOT/config/$f"
    [ -f "$SRC" ] || die "Config not found: $SRC"
    databricks fs cp "$SRC" "$CONFIG_DEST/$f" \
      --profile "$PROFILE" --overwrite \
      || die "Failed to stage $SRC -> $CONFIG_DEST/$f"
    echo "    Staged: $CONFIG_DEST/$f"
done

# Eval task JSONs (read from the Volume by the notebook).
for f in T1.json T2.json T3.json; do
    SRC="$REPO_ROOT/eval/tasks/$f"
    [ -f "$SRC" ] || die "Eval task file not found: $SRC"
    databricks fs cp "$SRC" "$EVAL_TASKS_DEST/$f" \
      --profile "$PROFILE" --overwrite \
      || die "Failed to stage $SRC -> $EVAL_TASKS_DEST/$f"
    echo "    Staged: $EVAL_TASKS_DEST/$f"
done

# ---------------------------------------------------------------------------
# 2. Pre-create artifact dirs for all (task, arm) combinations
#    Volumes do NOT auto-create nested dirs — pre-create before notebook runs.
# ---------------------------------------------------------------------------
echo "[2] Pre-creating artifact dirs for all 12 (task, arm) combinations..."
for task_id in T1 T2 T3; do
    for arm in empty static_generic retrieved placebo; do
        ART_DIR="${ARTIFACTS_DEST}/${task_id}/${arm}"
        databricks fs mkdir "$ART_DIR" --profile "$PROFILE" \
          || die "Failed to create artifact dir $ART_DIR"
        echo "    Ensured: $ART_DIR"
    done
done

# ---------------------------------------------------------------------------
# 3. Import the 04 notebook to the workspace
# ---------------------------------------------------------------------------
echo "[3] Importing notebook 04 to workspace..."
databricks workspace mkdirs "$WORKSPACE_PATH" --profile "$PROFILE" \
  || die "Failed to create workspace directory $WORKSPACE_PATH"
databricks workspace import "$WORKSPACE_PATH/04_compile_memory_md" \
  --file "$REPO_ROOT/notebooks/04_compile_memory_md.py" \
  --language PYTHON \
  --format SOURCE \
  --overwrite \
  --profile "$PROFILE"
echo "    Imported: $WORKSPACE_PATH/04_compile_memory_md"

# ---------------------------------------------------------------------------
# helpers: submit / wait / validate (mirror of run_embed.sh)
# ---------------------------------------------------------------------------

# submit_notebook_run <notebook_path> <run_name> <dry_run_param> <profile>
# Submits a serverless one-time run with the dry_run base-parameter and echoes run_id.
submit_notebook_run() {
    local NB_PATH="$1"
    local RUN_NAME="$2"
    local DRY_PARAM="$3"
    local PROFILE="$4"

    local PAYLOAD
    PAYLOAD=$(NB_PATH="$NB_PATH" RUN_NAME="$RUN_NAME" DRY_PARAM="$DRY_PARAM" python3 - <<'PY'
import json, os
payload = {
    "run_name": os.environ["RUN_NAME"],
    "tasks": [
        {
            "task_key": "main",
            "notebook_task": {
                "notebook_path": os.environ["NB_PATH"],
                "source":        "WORKSPACE",
                "base_parameters": {"dry_run": os.environ["DRY_PARAM"]},
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

# wait_for_run <run_id> <profile> — polls to a terminal state; exits 1 if not SUCCESS.
wait_for_run() {
    local RUN_ID="$1"
    local PROFILE="$2"
    local elapsed=0

    while [ $elapsed -lt $MAX_WAIT ]; do
        local LC_STATE
        LC_STATE=$(databricks jobs get-run "$RUN_ID" -o json --profile "$PROFILE" \
            | python3 -c "import json,sys; print(json.load(sys.stdin)['state']['life_cycle_state'])")
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
    echo ""

    if [ $elapsed -ge $MAX_WAIT ]; then
        die "Timed out waiting for run_id=$RUN_ID after ${MAX_WAIT}s"
    fi

    local RESULT_STATE
    RESULT_STATE=$(databricks jobs get-run "$RUN_ID" -o json --profile "$PROFILE" \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['state'].get('result_state','UNKNOWN'))")
    if [ "$RESULT_STATE" != "SUCCESS" ]; then
        echo "    RESULT: $RESULT_STATE"
        echo "    Inspect run: databricks jobs get-run $RUN_ID --profile $PROFILE"
        die "Notebook run failed (run_id=$RUN_ID result=$RESULT_STATE)"
    fi
    echo "    RESULT: $RESULT_STATE"
}

# validate_sentinel <parent_run_id> <expected_sentinel> <profile>
validate_sentinel() {
    local RUN_ID="$1"
    local EXPECTED="$2"
    local PROFILE="$3"

    local TASK_RUN_ID
    TASK_RUN_ID=$(databricks jobs get-run "$RUN_ID" -o json --profile "$PROFILE" \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['tasks'][0]['run_id'])")

    databricks jobs get-run-output "$TASK_RUN_ID" -o json --profile "$PROFILE" \
      | python3 "$REPO_ROOT/scripts/_check_sentinel.py" "$EXPECTED" \
      || die "Sentinel validation FAILED for run_id=$RUN_ID (task=$TASK_RUN_ID)"
}

# print_run_output <parent_run_id> <profile> — echoes the notebook exit result JSON.
print_run_output() {
    local RUN_ID="$1"
    local PROFILE="$2"
    local TASK_RUN_ID
    TASK_RUN_ID=$(databricks jobs get-run "$RUN_ID" -o json --profile "$PROFILE" \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['tasks'][0]['run_id'])")
    echo "    --- notebook exit output ---"
    databricks jobs get-run-output "$TASK_RUN_ID" -o json --profile "$PROFILE" \
      | python3 -c "
import json,sys
d=json.load(sys.stdin)
res=(d.get('notebook_output') or {}).get('result')
print('    ' + (res if res else '(no notebook_output.result)'))
"
}

# ---------------------------------------------------------------------------
# 4. Submit the run
# ---------------------------------------------------------------------------
echo "[4] Submitting compile run (mode=$MODE)..."
RUN_ID=$(submit_notebook_run \
    "$WORKSPACE_PATH/04_compile_memory_md" \
    "compile-$MODE" \
    "$DRY_RUN_PARAM" \
    "$PROFILE")
echo "    run_id=$RUN_ID — waiting for completion..."
wait_for_run "$RUN_ID" "$PROFILE"
validate_sentinel "$RUN_ID" "$EXPECTED_SENTINEL" "$PROFILE"
print_run_output "$RUN_ID" "$PROFILE"

echo ""
echo "=== Compile run ($MODE) completed and sentinel-validated ==="
echo "    expected sentinel : $EXPECTED_SENTINEL"
if [ "$MODE" = "dry_run" ]; then
    echo "    Dry run wrote NOTHING. Review the 12 file_sha256 hashes above"
    echo "    before running 'full' to freeze artifacts to the UC Volume."
fi
