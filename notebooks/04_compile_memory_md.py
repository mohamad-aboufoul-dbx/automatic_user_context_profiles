# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 — Compile Memory Artifacts → memory_artifacts
# MAGIC
# MAGIC For each `(task ∈ {T1,T2,T3}) × (arm ∈ {empty,static_generic,retrieved,placebo})`:
# MAGIC   1. Reads the compile corpus from `atomic_memories` (99 rows after metadata filter).
# MAGIC   2. Embeds the three task goal-texts via the serving endpoint in `compile_config.json`.
# MAGIC   3. Calls `compiler.arms.pool` + `compiler.render.render` to produce a deterministic
# MAGIC      `MEMORY.md` with `file_sha256`, `payload_sha256`, and `sentinel`.
# MAGIC   4. Asserts byte-for-byte determinism (re-compiles every artifact once more and
# MAGIC      checks identical `file_sha256`).
# MAGIC   5. (Full run only) Writes each `.md` to the UC Volume and MERGEs 12 rows into
# MAGIC      `memory_artifacts`.
# MAGIC
# MAGIC **This notebook MUST NOT run automatically.** Defaults to `dry_run="true"`
# MAGIC (safe: full computation + assertions, writes nothing). The controller flips
# MAGIC `dry_run="false"` for the actual freeze.
# MAGIC
# MAGIC ### Staging (agreed with `scripts/run_compile.sh`)
# MAGIC   - `compiler` package  → `/Volumes/.../raw/compiler_src/compiler/*.py`
# MAGIC     (parent dir `.../raw/compiler_src` is added to `sys.path`)
# MAGIC   - `extract` package   → `/Volumes/.../raw/extract_src/extract/*.py`
# MAGIC     (parent dir `.../raw/extract_src` is added to `sys.path`)
# MAGIC   - config JSONs        → `/Volumes/.../raw/config/*.json`
# MAGIC   - eval task JSONs     → `/Volumes/.../raw/eval_tasks/{T1,T2,T3}.json`
# MAGIC
# MAGIC ### Sentinel
# MAGIC   - dry-run: `{"sentinel":"compile_dryrun:OK","status":"OK","stage":"compile_dryrun",...}`
# MAGIC   - full:    `{"sentinel":"compile:OK","status":"OK","stage":"compile",...}`


# COMMAND ----------


import json
import sys
from datetime import datetime, timezone

from pyspark.sql.types import (
    ArrayType, IntegerType, StringType, StructField, StructType, TimestampType,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATALOG       = "ai_fde_hackathon_catalog"
SCHEMA        = "automatic_user_context_profiles"
ATOMIC_TBL    = f"{CATALOG}.{SCHEMA}.atomic_memories"
ARTIFACTS_TBL = f"{CATALOG}.{SCHEMA}.memory_artifacts"

VOLUME_ROOT          = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
EXTRACT_SRC_DIR      = f"{VOLUME_ROOT}/extract_src"        # parent of `extract` pkg
COMPILER_SRC_DIR     = f"{VOLUME_ROOT}/compiler_src"       # parent of `compiler` pkg
CONFIG_DIR           = f"{VOLUME_ROOT}/config"
EVAL_TASKS_DIR       = f"{VOLUME_ROOT}/eval_tasks"
ARTIFACTS_BASE_DIR   = f"{VOLUME_ROOT}/artifacts"

COMPILE_CONFIG_PATH     = f"{CONFIG_DIR}/compile_config.json"
HELDOUT_EXCLUSIONS_PATH = f"{CONFIG_DIR}/heldout_exclusions.json"

EXPECTED_DIM = 1024
ARMS         = ["empty", "static_generic", "retrieved", "placebo"]
TASK_IDS     = ["T1", "T2", "T3"]

# ---------------------------------------------------------------------------
# dry_run widget — DEFAULT "true" (safe). Only the literal "false" enables writes.
# ---------------------------------------------------------------------------
try:
    dbutils.widgets.text("dry_run", "true")   # noqa: F821
    _dry_raw = dbutils.widgets.get("dry_run")  # noqa: F821
except Exception:
    _dry_raw = "true"
DRY_RUN = _dry_raw.strip().lower() != "false"
print(f"dry_run param = {_dry_raw!r}  ->  DRY_RUN={DRY_RUN}")


# COMMAND ----------

# MAGIC %md ## Add staged source packages to sys.path

# COMMAND ----------

# Add both the extract and compiler package parents to sys.path so that
# `import extract.*` and `import compiler.*` resolve to the staged Volume copies.
for _src_dir in (EXTRACT_SRC_DIR, COMPILER_SRC_DIR):
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
        print(f"Added to sys.path: {_src_dir}")

from compiler.arms     import pool                        # noqa: E402
from compiler.render   import render                      # noqa: E402
from compiler.artifact import (                           # noqa: E402
    make_artifact_id,
    spark_row_to_mem,
)
from compiler          import COMPILER_VERSION            # noqa: E402
from extract.prefilter import unresolved_prefix_ids       # noqa: E402

print(f"COMPILER_VERSION = {COMPILER_VERSION!r}")


# COMMAND ----------

# MAGIC %md ## Load staged config + build the exclusion sets

# COMMAND ----------

with open(COMPILE_CONFIG_PATH, "r", encoding="utf-8") as _fh:
    CFG = json.load(_fh)
    _fh.seek(0)
    COMPILE_CONFIG_TEXT = _fh.read()

with open(HELDOUT_EXCLUSIONS_PATH, "r", encoding="utf-8") as _fh:
    EXCL_DOC = json.load(_fh)

HELDOUT_IDS: set[str] = set(CFG.get("heldout_conversation_ids", []))

CLUSTER_EXCLUSIONS: set[str] = set()
for _key, _val in EXCL_DOC.items():
    if isinstance(_val, dict) and "conversation_ids" in _val:
        CLUSTER_EXCLUSIONS.update(_val["conversation_ids"])

BANNED_IDS: set[str] = HELDOUT_IDS | CLUSTER_EXCLUSIONS

print(f"heldout ids (tight)  : {len(HELDOUT_IDS)}")
print(f"cluster exclusions   : {len(CLUSTER_EXCLUSIONS)}")
print(f"banned ids (union)   : {len(BANNED_IDS)}")
print(f"COMPILER_VERSION     : {COMPILER_VERSION}")

EMBED_ENDPOINT = CFG["embedding_model"]
print(f"embedding_model (endpoint): {EMBED_ENDPOINT}")

NOW = CFG["cutoff_ts"]
print(f"Evaluation reference timestamp (NOW): {NOW}")


# COMMAND ----------

# MAGIC %md ## Read atomic_memories — apply metadata filter

# COMMAND ----------

_mf          = CFG["metadata_filter"]
_username    = _mf["username"]
_domains     = _mf["domain_in"]
_cutoff_ts   = CFG["cutoff_ts"]

_domains_sql = ", ".join(f"'{d}'" for d in _domains)

_cutoff_dt_str = (
    _cutoff_ts.replace("T", " ").rstrip("Z")
)

df_corpus = spark.sql(f"""
    SELECT
        memory_id,
        memory_text,
        memory_type,
        domain,
        embedding,
        confidence,
        source_datetime,
        conversation_id
    FROM {ATOMIC_TBL}
    WHERE username = '{_username}'
      AND domain IN ({_domains_sql})
      AND source_datetime < TIMESTAMP '{_cutoff_dt_str}'
      AND embedding IS NOT NULL
""")   # noqa: F821

corpus_rows = df_corpus.collect()
n_corpus    = len(corpus_rows)
print(f"Corpus rows (after metadata filter, embedding IS NOT NULL): {n_corpus}")
if n_corpus == 0:
    raise RuntimeError(
        "CORPUS EMPTY: no rows returned by the metadata filter. "
        "Verify that atomic_memories is populated and embeddings are present."
    )


# COMMAND ----------

# MAGIC %md ## Contamination guard — prefix check + exact banned JOIN

# COMMAND ----------

_corpus_conv_ids: set[str] = {r["conversation_id"] for r in corpus_rows}

_unresolved = unresolved_prefix_ids(BANNED_IDS, _corpus_conv_ids)
if _unresolved:
    raise RuntimeError(
        "CONTAMINATION GUARD MISCONFIGURED: these banned ids are only a PREFIX "
        f"of a real conversation_id (exact-equality check will miss them): {_unresolved}. "
        "Resolve them to full conversation_ids in config/heldout_exclusions.json."
    )

_banned_in_corpus = [
    r["conversation_id"]
    for r in corpus_rows
    if r["conversation_id"] in BANNED_IDS
]
if _banned_in_corpus:
    raise RuntimeError(
        f"CONTAMINATION: {len(_banned_in_corpus)} corpus rows reference held-out/cluster "
        f"conversation_ids: {_banned_in_corpus[:5]}. Compile run is invalid."
    )

print("PASS: zero held-out/cluster contamination in corpus.")


# COMMAND ----------

# MAGIC %md ## Convert corpus rows to compiler-ready mem dicts (inject token_count)

# COMMAND ----------

all_mems: list[dict] = [spark_row_to_mem(r) for r in corpus_rows]
print(f"Converted {len(all_mems)} rows to compiler-ready dicts.")
print(f"Sample token counts: {[m['token_count'] for m in all_mems[:5]]}")


# COMMAND ----------

# MAGIC %md ## Load eval task definitions from staged Volume

# COMMAND ----------

task_defs: dict[str, dict] = {}
for _tid in TASK_IDS:
    _path = f"{EVAL_TASKS_DIR}/{_tid}.json"
    with open(_path, "r", encoding="utf-8") as _fh:
        task_defs[_tid] = json.load(_fh)
    print(f"Loaded task {_tid}: {_path}")


# COMMAND ----------

# MAGIC %md ## WorkspaceClient + embed_batch helper (mirrors notebook 03)

# COMMAND ----------

from databricks.sdk import WorkspaceClient   # noqa: E402

_WS = WorkspaceClient()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Call the embedding serving endpoint with one batch; return vectors in input order.

    - Defensively sorts by d.index if the attribute is present.
    - Asserts exactly len(texts) vectors returned.
    - Casts every component to float.
    - Asserts every vector has EXPECTED_DIM components.
    """
    resp = _WS.serving_endpoints.query(name=EMBED_ENDPOINT, input=texts)
    data = list(resp.data or [])
    if data and hasattr(data[0], "index"):
        data = sorted(data, key=lambda d: d.index)
    vectors = [[float(x) for x in d.embedding] for d in data]
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedding endpoint returned {len(vectors)} vectors for {len(texts)} inputs"
        )
    for i, vec in enumerate(vectors):
        if len(vec) != EXPECTED_DIM:
            raise RuntimeError(
                f"Vector[{i}] has dimension {len(vec)}, expected {EXPECTED_DIM} "
                f"(endpoint={EMBED_ENDPOINT!r})."
            )
    return vectors


# COMMAND ----------

# MAGIC %md ## Embed the three goal-prompt texts (one batch; cache per run)

# COMMAND ----------

_goal_texts = [task_defs[tid]["goal_prompt"] for tid in TASK_IDS]
print(f"Embedding {len(_goal_texts)} goal texts in a single batch...")
_goal_vecs_list = embed_batch(_goal_texts)

goal_embeddings: dict[str, list[float]] = {
    tid: _goal_vecs_list[i]
    for i, tid in enumerate(TASK_IDS)
}
for tid, vec in goal_embeddings.items():
    print(f"  {tid}: dim={len(vec)}  first5={vec[:5]}")


# COMMAND ----------

# MAGIC %md ## Build task dicts (one per task — goal_embedding + repo_domain)

# COMMAND ----------

task_dicts: dict[str, dict] = {}
for tid in TASK_IDS:
    task_dicts[tid] = {
        "goal_embedding": goal_embeddings[tid],
        "repo_domain":    "repo:platform",
    }
print("Task dicts built for:", list(task_dicts.keys()))


# COMMAND ----------

# MAGIC %md ## Compile all 12 (task × arm) artifacts

# COMMAND ----------

artifacts: list[dict] = []
for tid in TASK_IDS:
    _task = task_dicts[tid]
    for arm in ARMS:
        _mems  = pool(arm, all_mems, _task, CFG, NOW)
        _res   = render(tid, arm, _mems, CFG)
        artifacts.append({
            "task_id":  tid,
            "arm":      arm,
            "mems":     _mems,
            "result":   _res,
        })
        print(
            f"  {tid}/{arm:15s} "
            f"selected={len(_mems):2d}  "
            f"tokens={_res.token_count:4d}  "
            f"file_sha256={_res.file_sha256[:16]}..."
        )

print(f"\nTotal artifacts compiled: {len(artifacts)}")


# COMMAND ----------

# MAGIC %md ## Post-selection contamination re-assert

# COMMAND ----------

for _art in artifacts:
    for _m in _art["mems"]:
        if _m["conversation_id"] in BANNED_IDS:
            raise RuntimeError(
                f"CONTAMINATION: memory {_m['memory_id']!r} from banned conversation "
                f"{_m['conversation_id']!r} was selected by pool() for "
                f"({_art['task_id']}, {_art['arm']}). Compile run is invalid."
            )

print("PASS: zero held-out/cluster mems in any selected pool.")


# COMMAND ----------

# MAGIC %md ## Determinism assertion — re-compile all 12, assert byte-for-byte equality

# COMMAND ----------

for _art in artifacts:
    _task2 = task_dicts[_art["task_id"]]
    _mems2 = pool(_art["arm"], all_mems, _task2, CFG, NOW)
    _res2  = render(_art["task_id"], _art["arm"], _mems2, CFG)
    if _res2.file_sha256 != _art["result"].file_sha256:
        raise RuntimeError(
            f"DETERMINISM ASSERTION FAILED: ({_art['task_id']}, {_art['arm']}) "
            f"first={_art['result'].file_sha256!r} second={_res2.file_sha256!r}"
        )

print("PASS: all 12 artifacts are byte-for-byte deterministic across two compile passes.")


# COMMAND ----------

# MAGIC %md ## Count assertion — exactly 12 artifacts

# COMMAND ----------

if len(artifacts) != 12:
    raise RuntimeError(
        f"ARTIFACT COUNT ASSERTION FAILED: expected 12, got {len(artifacts)}. "
        f"Expected 3 tasks × 4 arms."
    )

for _art in artifacts:
    if not _art["result"].file_sha256:
        raise RuntimeError(
            f"EMPTY SHA: ({_art['task_id']}, {_art['arm']}) has empty file_sha256."
        )

print(f"PASS: exactly 12 artifacts, all with non-empty file_sha256.")


# COMMAND ----------

# MAGIC %md ## DRY-RUN gate — print all 12 shas, write NOTHING, exit

# COMMAND ----------

_sha_summary = [
    {"task_id": a["task_id"], "arm": a["arm"], "file_sha256": a["result"].file_sha256}
    for a in artifacts
]

if DRY_RUN:
    print("=== DRY RUN — all assertions passed; writing NOTHING ===")
    for _s in _sha_summary:
        print(f"  {_s['task_id']}/{_s['arm']:15s}: {_s['file_sha256']}")

    _sentinel_payload = {
        "sentinel":    "compile_dryrun:OK",
        "status":      "OK",
        "stage":       "compile_dryrun",
        "n_artifacts": len(artifacts),
        "n_corpus":    n_corpus,
        "sha256s":     _sha_summary,
    }
    dbutils.notebook.exit(json.dumps(_sentinel_payload))   # noqa: F821


# COMMAND ----------

# MAGIC %md ## FULL RUN — write .md files to UC Volume + MERGE 12 rows

# COMMAND ----------

# (Only reached when DRY_RUN is False — dry-run exits above.)

FROZEN_AT = datetime.now(timezone.utc)
print(f"frozen_at = {FROZEN_AT.isoformat()}")

written_files: list[str] = []

for _art in artifacts:
    _tid    = _art["task_id"]
    _arm    = _art["arm"]
    _md     = _art["result"].markdown

    _dir_dbfs = f"dbfs:{ARTIFACTS_BASE_DIR}/{_tid}/{_arm}"
    dbutils.fs.mkdirs(_dir_dbfs)   # noqa: F821

    _file_dbfs    = f"{_dir_dbfs}/MEMORY.md"
    _file_vol_path = f"{ARTIFACTS_BASE_DIR}/{_tid}/{_arm}/MEMORY.md"

    dbutils.fs.put(_file_dbfs, _md, overwrite=True)   # noqa: F821

    _art["volume_path"] = _file_vol_path
    written_files.append(_file_vol_path)
    print(f"  Wrote: {_file_vol_path}  ({len(_md)} chars)")

print(f"Wrote {len(written_files)} MEMORY.md files to the UC Volume.")


# COMMAND ----------

# MAGIC %md ## MERGE 12 rows into memory_artifacts

# COMMAND ----------

ART_SCHEMA = StructType([
    StructField("artifact_id",         StringType(),            False),
    StructField("task_id",             StringType(),            False),
    StructField("arm",                 StringType(),            False),
    StructField("compiler_version",    StringType(),            False),
    StructField("config_json",         StringType(),            False),
    StructField("selected_memory_ids", ArrayType(StringType()), False),
    StructField("payload_sha256",      StringType(),            False),
    StructField("file_sha256",         StringType(),            False),
    StructField("sentinel",            StringType(),            False),
    StructField("memory_markdown",     StringType(),            False),
    StructField("token_count",         IntegerType(),           False),
    StructField("volume_path",         StringType(),            False),
    StructField("frozen_at",           TimestampType(),         False),
])

art_rows = []
for _art in artifacts:
    _res = _art["result"]
    _artifact_id = make_artifact_id(_art["task_id"], _art["arm"], _res.file_sha256)
    art_rows.append((
        _artifact_id,
        _art["task_id"],
        _art["arm"],
        COMPILER_VERSION,
        COMPILE_CONFIG_TEXT,
        [m["memory_id"] for m in _art["mems"]],
        _res.payload_sha256,
        _res.file_sha256,
        _res.sentinel,
        _res.markdown,
        int(_res.token_count),
        _art["volume_path"],
        FROZEN_AT,
    ))

df_art = spark.createDataFrame(art_rows, schema=ART_SCHEMA)   # noqa: F821
df_art.createOrReplaceTempView("_new_artifacts")

spark.sql(f"""
    MERGE INTO {ARTIFACTS_TBL} t
    USING _new_artifacts s
    ON t.task_id = s.task_id AND t.arm = s.arm
    WHEN MATCHED THEN UPDATE SET
        t.artifact_id         = s.artifact_id,
        t.task_id             = s.task_id,
        t.arm                 = s.arm,
        t.compiler_version    = s.compiler_version,
        t.config_json         = s.config_json,
        t.selected_memory_ids = s.selected_memory_ids,
        t.payload_sha256      = s.payload_sha256,
        t.file_sha256         = s.file_sha256,
        t.sentinel            = s.sentinel,
        t.memory_markdown     = s.memory_markdown,
        t.token_count         = s.token_count,
        t.volume_path         = s.volume_path,
        t.frozen_at           = s.frozen_at
    WHEN NOT MATCHED THEN INSERT (
        artifact_id, task_id, arm, compiler_version, config_json,
        selected_memory_ids, payload_sha256, file_sha256, sentinel,
        memory_markdown, token_count, volume_path, frozen_at
    ) VALUES (
        s.artifact_id, s.task_id, s.arm, s.compiler_version, s.config_json,
        s.selected_memory_ids, s.payload_sha256, s.file_sha256, s.sentinel,
        s.memory_markdown, s.token_count, s.volume_path, s.frozen_at
    )
""")   # noqa: F821

print(f"MERGE complete — {len(art_rows)} rows upserted into {ARTIFACTS_TBL}.")


# COMMAND ----------

# MAGIC %md ## Post-write assertion — verify row count in memory_artifacts

# COMMAND ----------

_ta_rows   = [(_art["task_id"], _art["arm"]) for _art in artifacts]
_ta_schema = StructType([
    StructField("task_id", StringType(), False),
    StructField("arm",     StringType(), False),
])
_ta_df = spark.createDataFrame(_ta_rows, schema=_ta_schema)   # noqa: F821
_ta_df.createOrReplaceTempView("_expected_task_arms")

_found = spark.sql(f"""
    SELECT COUNT(*) AS n
    FROM {ARTIFACTS_TBL} t
    JOIN _expected_task_arms e
      ON t.task_id = e.task_id AND t.arm = e.arm
""").first()["n"]   # noqa: F821

if _found != 12:
    raise RuntimeError(
        f"POST-MERGE ASSERTION FAILED: expected 12 rows in memory_artifacts "
        f"matching our (task_id, arm) pairs, found {_found}."
    )
print(f"PASS: {_found} rows confirmed in {ARTIFACTS_TBL}.")


# COMMAND ----------

_sentinel_payload = {
    "sentinel":    "compile:OK",
    "status":      "OK",
    "stage":       "compile",
    "n_artifacts": len(artifacts),
    "n_corpus":    n_corpus,
    "count":       12,
    "sha256s":     _sha_summary,
}
dbutils.notebook.exit(json.dumps(_sentinel_payload))   # noqa: F821
