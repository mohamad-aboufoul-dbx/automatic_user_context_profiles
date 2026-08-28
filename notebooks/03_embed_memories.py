# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 — Embed Atomic Memories
# MAGIC
# MAGIC Fills the `embedding` and `embedding_model` columns in
# MAGIC `ai_fde_hackathon_catalog.automatic_user_context_profiles.atomic_memories`
# MAGIC for every row where `embedding IS NULL`. Uses the Databricks embedding serving
# MAGIC endpoint configured in `compile_config.json` (`embedding_model` key).
# MAGIC
# MAGIC **This notebook MUST NOT run automatically.** It defaults to `dry_run="true"`
# MAGIC (safe: one embedding call, writes nothing). The controller flips `dry_run="false"`.
# MAGIC
# MAGIC ### Embedding
# MAGIC   - Endpoint: read from `config/compile_config.json` key `embedding_model`
# MAGIC   - Expected dimension: 1024 (GTE-large-en); asserted for every returned vector
# MAGIC   - Rows are embedded in batches (`EMBED_BATCH = 64`)
# MAGIC   - Written back via MERGE on `memory_id` (UPDATE SET only — rows already exist)
# MAGIC
# MAGIC ### Idempotency
# MAGIC   - Only rows with `embedding IS NULL` are selected; re-running embeds 0 rows
# MAGIC     once all columns are filled.
# MAGIC
# MAGIC ### Optional dedup (SPEC §4, default OFF)
# MAGIC   - `CFG.get("dedup_enabled", False)` guards destructive deletes
# MAGIC   - When OFF: candidate near-duplicate pairs (cosine ≥ 0.95 within a `memory_type`)
# MAGIC     are logged for visibility but NOT deleted
# MAGIC
# MAGIC ### Staging (agreed with `scripts/run_embed.sh`)
# MAGIC   - guard modules → `/Volumes/ai_fde_hackathon_catalog/automatic_user_context_profiles/raw/extract_src/extract/*.py`
# MAGIC     (parent dir `.../raw/extract_src` is added to `sys.path`)
# MAGIC   - config JSONs → `/Volumes/ai_fde_hackathon_catalog/automatic_user_context_profiles/raw/config/*.json`
# MAGIC
# MAGIC ### Sentinel
# MAGIC The runner validates `dbutils.notebook.exit` via `scripts/_check_sentinel.py`,
# MAGIC which reads the `"sentinel"` field. We therefore emit a `"sentinel"` key
# MAGIC ALONGSIDE the task-specified status payload:
# MAGIC   - dry-run: `{"sentinel":"embed_dryrun:OK","status":"OK","stage":"embed_dryrun",...}`
# MAGIC   - full:    `{"sentinel":"embed:OK","status":"OK","stage":"embed",...}`


# COMMAND ----------


import json
import math
import sys
from collections import defaultdict

from pyspark.sql.types import (
    ArrayType, FloatType, StringType, StructField, StructType,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATALOG    = "ai_fde_hackathon_catalog"
SCHEMA     = "automatic_user_context_profiles"
ATOMIC_TBL = f"{CATALOG}.{SCHEMA}.atomic_memories"

VOLUME_ROOT             = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
EXTRACT_SRC_DIR         = f"{VOLUME_ROOT}/extract_src"       # parent of the `extract` package
CONFIG_DIR              = f"{VOLUME_ROOT}/config"             # staged compile_config + exclusions
COMPILE_CONFIG_PATH     = f"{CONFIG_DIR}/compile_config.json"
HELDOUT_EXCLUSIONS_PATH = f"{CONFIG_DIR}/heldout_exclusions.json"

EXPECTED_DIM = 1024      # GTE-large-en embedding width — asserted for every vector
EMBED_BATCH  = 64        # rows per serving-endpoint call

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
# EXTRACT_SRC_DIR/extract/*.py. Add the PARENT to sys.path. We do not need
# prefilter/ids for embedding itself, but the package must be resolvable for
# consistency and the contamination re-assert (BANNED_IDS is built from config
# files; no `extract.*` import is required here).
if EXTRACT_SRC_DIR not in sys.path:
    sys.path.insert(0, EXTRACT_SRC_DIR)

print(f"Added to sys.path: {EXTRACT_SRC_DIR}")


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

print(f"heldout ids (tight)  : {len(HELDOUT_IDS)}")
print(f"cluster exclusions   : {len(CLUSTER_EXCLUSIONS)}")
print(f"banned ids (union)   : {len(BANNED_IDS)}")

# Embedding endpoint: read from config (do NOT hardcode a different value).
EMBED_ENDPOINT = CFG["embedding_model"]
print(f"embedding_model (endpoint): {EMBED_ENDPOINT}")


# COMMAND ----------

# MAGIC %md ## Read null-embedding rows from atomic_memories

# COMMAND ----------

df_to_embed = spark.sql(f"""  # noqa: F821
    SELECT memory_id, memory_text
    FROM {ATOMIC_TBL}
    WHERE embedding IS NULL
""")
to_embed = df_to_embed.collect()
n_to_embed = len(to_embed)
print(f"Rows with embedding IS NULL: {n_to_embed}")


# COMMAND ----------

# MAGIC %md ## WorkspaceClient + embedding helper

# COMMAND ----------

from databricks.sdk import WorkspaceClient   # noqa: E402

_WS = WorkspaceClient()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Call the embedding serving endpoint with one batch; return vectors in input order.

    - Defensively sorts by `d.index` if the attribute is present (preserves
      input→output order even when the endpoint reorders results).
    - Asserts that the endpoint returned exactly `len(texts)` vectors.
    - Casts every component to `float` (target column is ARRAY<FLOAT>).
    - Asserts every vector has `EXPECTED_DIM` components; raises loudly if not.
    """
    resp = _WS.serving_endpoints.query(name=EMBED_ENDPOINT, input=texts)
    data = list(resp.data or [])
    # Sort by .index if the attribute is present (defensive ordering guarantee).
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
                f"(endpoint={EMBED_ENDPOINT!r}). Refusing to write wrong-width vectors."
            )
    return vectors


# COMMAND ----------

# MAGIC %md ## DRY-RUN: sample up to 3 rows, print vector info, write NOTHING

# COMMAND ----------

if DRY_RUN:
    print("=== DRY RUN — sample embedding call, no writes ===")
    sample_dim = 0
    if not to_embed:
        print("No rows with embedding IS NULL — nothing to embed.")
    else:
        sample_rows = to_embed[:3]
        sample_texts = [r["memory_text"] for r in sample_rows]
        sample_vecs = embed_batch(sample_texts)
        sample_dim = len(sample_vecs[0])
        for row, vec in zip(sample_rows, sample_vecs):
            print(f"memory_id : {row['memory_id']}")
            print(f"text      : {row['memory_text'][:120]!r}")
            print(f"vec len   : {len(vec)}")
            print(f"first 5   : {vec[:5]}")
            print()

    sentinel_payload = {
        "sentinel":   "embed_dryrun:OK",
        "status":     "OK",
        "stage":      "embed_dryrun",
        "to_embed":   n_to_embed,
        "sample_dim": sample_dim,
        "endpoint":   EMBED_ENDPOINT,
    }
    dbutils.notebook.exit(json.dumps(sentinel_payload))   # noqa: F821


# COMMAND ----------

# MAGIC %md ## FULL RUN: embed every null row in batches, MERGE back, assert

# COMMAND ----------

# (Only reached when DRY_RUN is False — dry-run exits above.)

# --- Embed all null-embedding rows in batches ---
ids_texts = [(r["memory_id"], r["memory_text"]) for r in to_embed]
embed_results: list[tuple[str, list[float]]] = []   # (memory_id, vector)

for batch_start in range(0, len(ids_texts), EMBED_BATCH):
    batch       = ids_texts[batch_start : batch_start + EMBED_BATCH]
    batch_ids   = [x[0] for x in batch]
    batch_texts = [x[1] for x in batch]
    batch_vecs  = embed_batch(batch_texts)
    for mid, vec in zip(batch_ids, batch_vecs):
        embed_results.append((mid, vec))
    print(f"    Embedded batch {batch_start // EMBED_BATCH + 1}: {len(batch)} rows")

embedded_count = len(embed_results)
print(f"Total rows embedded: {embedded_count}")

# --- Build DataFrame for MERGE ---
EMB_SCHEMA = StructType([
    StructField("memory_id",       StringType(),           False),
    StructField("embedding",       ArrayType(FloatType()), False),
    StructField("embedding_model", StringType(),           False),
])

embed_rows = [(mid, vec, EMBED_ENDPOINT) for mid, vec in embed_results]

if embed_rows:
    df_emb = spark.createDataFrame(embed_rows, schema=EMB_SCHEMA)   # noqa: F821
    df_emb.createOrReplaceTempView("_new_embeddings")
    spark.sql(f"""  # noqa: F821
        MERGE INTO {ATOMIC_TBL} t
        USING _new_embeddings s
        ON t.memory_id = s.memory_id
        WHEN MATCHED THEN UPDATE SET
            t.embedding       = s.embedding,
            t.embedding_model = s.embedding_model
    """)
    print(f"MERGE complete — {embedded_count} rows updated with embeddings.")
else:
    print("No embeddings produced — nothing to MERGE.")


# COMMAND ----------

# MAGIC %md ## Post-write assertions

# COMMAND ----------

# 1. No remaining NULLs (must be 0 after embed, regardless of whether dedup will run).
remaining_null = spark.sql(f"""  # noqa: F821
    SELECT COUNT(*) AS n FROM {ATOMIC_TBL} WHERE embedding IS NULL
""").first()["n"]
print(f"Remaining embedding IS NULL: {remaining_null}")
if remaining_null != 0:
    raise RuntimeError(
        f"POST-EMBED ASSERTION FAILED: {remaining_null} rows still have embedding IS NULL "
        "after the MERGE. Embedding run is incomplete."
    )
print("PASS: all rows now have embeddings.")

# 2. All non-null vectors have exactly EXPECTED_DIM components.
dim_rows = spark.sql(f"""  # noqa: F821
    SELECT DISTINCT size(embedding) AS dim
    FROM {ATOMIC_TBL}
    WHERE embedding IS NOT NULL
""").collect()
dims = [r["dim"] for r in dim_rows]
print(f"Distinct embedding dimensions in table: {dims}")
if dims != [EXPECTED_DIM]:
    raise RuntimeError(
        f"DIMENSION ASSERTION FAILED: expected exactly [{EXPECTED_DIM}], got {dims}"
    )
print(f"PASS: all vectors are {EXPECTED_DIM}-dimensional.")


# COMMAND ----------

# MAGIC %md ## Optional dedup (SPEC §4 — guarded, default OFF)

# COMMAND ----------

DEDUP_ENABLED = bool(CFG.get("dedup_enabled", False))
print(f"dedup_enabled = {DEDUP_ENABLED}  (config key absent → OFF)")

dropped_dupes: list[str] = []


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity (pure Python, O(dim)); returns 0 if either vector is zero."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _choose_keeper(ri, rj):
    """Return (keep_row, drop_row) for a near-duplicate pair.

    Priority: highest confidence → higher source_datetime → lex-smallest memory_id.
    """
    ci, cj = float(ri["confidence"]), float(rj["confidence"])
    di, dj = ri["source_datetime"], rj["source_datetime"]
    ii, ij = ri["memory_id"], rj["memory_id"]
    if ci != cj:
        return (ri, rj) if ci > cj else (rj, ri)
    if di != dj:
        return (ri, rj) if di > dj else (rj, ri)
    # Tie on both: lex-smallest memory_id is the keeper
    return (ri, rj) if ii <= ij else (rj, ri)


# Collect all embedded rows for dedup analysis (small set, O(n²) is acceptable).
df_all = spark.sql(f"""  # noqa: F821
    SELECT memory_id, memory_type, confidence, source_datetime, embedding
    FROM {ATOMIC_TBL}
    WHERE embedding IS NOT NULL
""")
all_rows = df_all.collect()

# Group by memory_type for within-type pair search.
groups: dict[str, list] = defaultdict(list)
for r in all_rows:
    groups[r["memory_type"]].append(r)

candidate_pairs: list[tuple[str, str, float]] = []   # (keep_id, drop_id, cosine)

for mtype, mrows in groups.items():
    for i in range(len(mrows)):
        for j in range(i + 1, len(mrows)):
            ri, rj = mrows[i], mrows[j]
            sim = _cosine(list(ri["embedding"]), list(rj["embedding"]))
            if sim >= 0.95:
                keep_r, drop_r = _choose_keeper(ri, rj)
                candidate_pairs.append((keep_r["memory_id"], drop_r["memory_id"], sim))

if candidate_pairs:
    print(f"Near-duplicate candidate pairs (cosine ≥ 0.95): {len(candidate_pairs)}")
    for keep_id, drop_id, sim in candidate_pairs:
        print(f"    cosine={sim:.4f}  KEEP={keep_id}  DROP={drop_id}")
else:
    print("No near-duplicate pairs found (cosine ≥ 0.95 threshold, within memory_type).")

if DEDUP_ENABLED:
    # Unique drop ids (a row may appear as a drop candidate in multiple pairs).
    to_drop_ids = list({drop_id for _, drop_id, _ in candidate_pairs})
    if to_drop_ids:
        drop_df = spark.createDataFrame(   # noqa: F821
            [(mid,) for mid in to_drop_ids],
            schema=StructType([StructField("memory_id", StringType(), False)]),
        )
        drop_df.createOrReplaceTempView("_dedup_drop_ids")
        spark.sql(f"""  # noqa: F821
            DELETE FROM {ATOMIC_TBL}
            WHERE memory_id IN (SELECT memory_id FROM _dedup_drop_ids)
        """)
        dropped_dupes = to_drop_ids
        for mid in dropped_dupes:
            print(f"    DROPPED (dedup): {mid}")
        print(f"Dedup: deleted {len(dropped_dupes)} near-duplicate rows.")
    else:
        print("Dedup enabled but no duplicates found — nothing deleted.")
else:
    print("Dedup is OFF — no rows deleted (see candidate pairs logged above, if any).")


# COMMAND ----------

# MAGIC %md ## Contamination re-assert — ZERO held-out/cluster rows in atomic_memories

# COMMAND ----------

# Build the banned-id set as a temp view and count matches by join (no string
# interpolation of ids into SQL). MUST be exactly 0.
banned_df = spark.createDataFrame(   # noqa: F821
    [(cid,) for cid in sorted(BANNED_IDS)],
    schema=StructType([StructField("conversation_id", StringType(), False)]),
)
banned_df.createOrReplaceTempView("_banned_conversation_ids")

heldout_rows = spark.sql(f"""  # noqa: F821
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
        "conversation_ids. Embedding run is invalid — refusing to certify this run."
    )
print("PASS: zero held-out/cluster contamination in atomic_memories.")


# COMMAND ----------

# Sentinel: must be the last thing that executes. The runner validates the
# "sentinel" field via scripts/_check_sentinel.py.
sentinel_payload = {
    "sentinel":       "embed:OK",
    "status":         "OK",
    "stage":          "embed",
    "embedded":       embedded_count,
    "remaining_null": remaining_null,
    "dim":            EXPECTED_DIM,
    "dedup_enabled":  DEDUP_ENABLED,
    "dropped_dupes":  len(dropped_dupes),
    "heldout_rows":   heldout_rows,
    "endpoint":       EMBED_ENDPOINT,
}
dbutils.notebook.exit(json.dumps(sentinel_payload))   # noqa: F821
