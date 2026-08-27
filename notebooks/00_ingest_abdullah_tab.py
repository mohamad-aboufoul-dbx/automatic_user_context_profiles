# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 00 — Ingest Abdullah Tab → raw_conversations_abdullah_said
# MAGIC
# MAGIC Loads the 'Abdullah' tab of the 'Chatbot conversations' data
# MAGIC (spreadsheet_id=1NCFFYeCPs-TS3YA6r3Y9y8p0LtUy2cY0-qa8CMl1iEI, gid=1354471250)
# MAGIC into UC table raw_conversations_abdullah_said via idempotent MERGE on row_hash.
# MAGIC
# MAGIC Source selection:
# MAGIC   - DEFAULT (INGEST_SOURCE=frozen_csv): the frozen committed CSV staged in
# MAGIC     the UC Volume — the reviewed, reproducible snapshot. Authoritative.
# MAGIC   - OPT-IN (INGEST_SOURCE=sheets): deliberately re-pull from the LIVE
# MAGIC     Google Sheet. Loudly announced; not reproducible. There is NO silent
# MAGIC     preference for the mutable Sheet.
# MAGIC
# MAGIC Invariants asserted (all hard failures):
# MAGIC   - Source row count in [240, 260] (~247 expected)
# MAGIC   - All rows username='abdullah.said' after filter
# MAGIC   - No unparseable datetimes
# MAGIC   - No empty conversation_ids
# MAGIC   - Source row_hashes unique after dedup
# MAGIC   - Post-MERGE row_hashes unique in target table


# COMMAND ----------


import hashlib
from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, TimestampType
)

CATALOG         = "ai_fde_hackathon_catalog"
SCHEMA          = "automatic_user_context_profiles"
TABLE           = f"{CATALOG}.{SCHEMA}.raw_conversations_abdullah_said"
SOURCE_NAME     = "chatbot_conversations_abdullah"
SPREADSHEET     = "1NCFFYeCPs-TS3YA6r3Y9y8p0LtUy2cY0-qa8CMl1iEI"
RANGE_NAME      = "Abdullah"
# The frozen, committed CSV — staged into a UC Volume by run_day1.sh — is the
# AUTHORITATIVE Day-1 source. It is the reviewed snapshot (~247 rows); using it
# by default makes runs reproducible and prevents a later edit to the live Sheet
# from silently replacing the reviewed data while still passing the count/user
# guards. (Public DBFS root is disabled here, so the Volume is the staging area.)
CSV_VOLUME      = "/Volumes/ai_fde_hackathon_catalog/automatic_user_context_profiles/raw/raw_abdullah_tab.csv"
USERNAME_FILTER = "abdullah.said"

# Source selection — DEFAULT is the frozen committed CSV (reproducible).
# Live Google Sheets is OPT-IN only: set widget/param INGEST_SOURCE="sheets"
# to deliberately re-pull from the (mutable) Sheet. Any other value → frozen CSV.
try:
    dbutils.widgets.text("INGEST_SOURCE", "frozen_csv")  # noqa: F821 (Databricks-injected)
    INGEST_SOURCE = dbutils.widgets.get("INGEST_SOURCE").strip().lower()  # noqa: F821
except Exception:
    INGEST_SOURCE = "frozen_csv"
USE_SHEETS = (INGEST_SOURCE == "sheets")

# Expected row count bounds (hard assertion).
# Adjust only if the source sheet is intentionally extended.
COUNT_LO = 240
COUNT_HI = 260


# COMMAND ----------

# MAGIC %md ## 1. Fetch raw rows (frozen Volume CSV by default; Sheets is opt-in)

# COMMAND ----------

def fetch_from_sheets() -> list[dict]:
    """Fetch Abdullah tab from Google Sheets using ADC token."""
    import urllib.request, json, subprocess
    token = subprocess.check_output(
        ["gcloud", "auth", "application-default", "print-access-token"],
        stderr=subprocess.DEVNULL,
    ).decode().strip()
    if not token:
        raise RuntimeError("Empty token from gcloud ADC")
    url = (
        f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET}"
        f"/values/{RANGE_NAME}"
    )
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "x-goog-user-project": "gcp-dev-field-eng-aiapiquota",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    rows_raw = data.get("values", [])
    if not rows_raw:
        raise ValueError("Empty response from Sheets API")
    header  = rows_raw[0]
    records = []
    for row in rows_raw[1:]:
        padded = row + [""] * (len(header) - len(row))
        records.append(dict(zip(header, padded[:len(header)])))
    return records


def fetch_from_csv(path: str) -> list[dict]:
    """Load CSV produced by the Google Sheets export."""
    import csv
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(dict(row))
    return records


records: list[dict] | None = None
source_path = None
errors: list[str] = []

if USE_SHEETS:
    # OPT-IN: deliberately re-pull from the live (mutable) Google Sheet.
    # Loudly announced so a Sheets-sourced run is never mistaken for the frozen
    # reproducible snapshot. No silent fallback to CSV here — if an explicit
    # Sheets pull is requested and fails, fail hard rather than quietly swapping
    # in a different source.
    print("=" * 70)
    print("INGEST_SOURCE=sheets — pulling from the LIVE Google Sheet (OPT-IN).")
    print("This is NOT the frozen reviewed snapshot and is not reproducible.")
    print("=" * 70)
    records = fetch_from_sheets()
    source_path = f"sheets_api:{SPREADSHEET}/{RANGE_NAME}"
    print(f"Sheets API: {len(records)} rows")
else:
    # DEFAULT: frozen committed CSV staged in the UC Volume — authoritative,
    # reproducible, reviewed snapshot.
    print(f"INGEST_SOURCE=frozen_csv (default) — using reviewed snapshot {CSV_VOLUME}")
    try:
        records = fetch_from_csv(CSV_VOLUME)
        source_path = CSV_VOLUME
        print(f"Frozen CSV: {len(records)} rows")
    except Exception as exc:
        errors.append(f"CSV {CSV_VOLUME}: {exc}")

if records is None:
    raise RuntimeError(
        "Ingest source failed (no silent fallback):\n" + "\n".join(errors)
    )

print(f"Source: {source_path}  |  raw rows: {len(records)}")


# COMMAND ----------

# MAGIC %md ## 2. Conform + hash + validate

# COMMAND ----------

DATETIME_FMTS = [
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
]


def parse_dt(s: str) -> datetime | None:
    for fmt in DATETIME_FMTS:
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def make_row_hash(
    username: str, query_text: str, response_text: str,
    chat_step: int, conversation_id: str, source_tool: str,
    event_datetime_iso: str,
) -> str:
    """SHA-256 of US-unit-separator-delimited canonical fields."""
    canonical = "\x1f".join([
        str(username        or "").strip().lower(),
        str(query_text      or "").strip(),
        str(response_text   or "").strip(),
        str(chat_step       or ""),
        str(conversation_id or "").strip(),
        str(source_tool     or "").strip(),
        event_datetime_iso,
    ])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


ingested_at = datetime.now(timezone.utc)
conformed: list[dict] = []
skipped_non_user = 0

for i, rec in enumerate(records):
    username = (rec.get("Username") or "").strip()
    if username != USERNAME_FILTER:
        skipped_non_user += 1
        continue

    query_text      = rec.get("Query",          "") or ""
    response_text   = rec.get("Response",       "") or ""
    chat_step_raw   = rec.get("Chat_step",      "0") or "0"
    conversation_id = (rec.get("Conversation_id") or "").strip()
    source_tool     = (rec.get("Tool",          "") or "").strip()
    dt_str          = (rec.get("Datetime",      "") or "").strip()

    # Hard assertion: datetime must be parseable
    assert dt_str, (
        f"Row {i}: empty Datetime field for conversation_id={conversation_id!r}"
    )
    event_dt = parse_dt(dt_str)
    assert event_dt is not None, (
        f"Row {i}: unparseable Datetime {dt_str!r} for conversation_id={conversation_id!r}"
    )

    # Hard assertion: conversation_id must be non-empty
    assert conversation_id, f"Row {i}: empty Conversation_id (Datetime={dt_str!r})"

    try:
        chat_step_int = int(float(chat_step_raw))
    except (ValueError, TypeError):
        raise AssertionError(
            f"Row {i}: unparseable Chat_step {chat_step_raw!r} "
            f"for conversation_id={conversation_id!r}"
        )

    h = make_row_hash(
        username, query_text, response_text,
        chat_step_int, conversation_id, source_tool,
        event_dt.isoformat(),
    )
    conformed.append({
        "username":         username,
        "query_text":       query_text,
        "response_text":    response_text,
        "chat_step":        chat_step_int,
        "conversation_id":  conversation_id,
        "source_tool":      source_tool,
        "event_datetime":   event_dt,
        "ingested_at":      ingested_at,
        "source_name":      SOURCE_NAME,
        "row_hash":         h,
    })

print(f"Conformed rows           : {len(conformed)}")
print(f"Skipped (non-{USERNAME_FILTER}): {skipped_non_user}")

# Hard assertions on conformed data
assert COUNT_LO <= len(conformed) <= COUNT_HI, (
    f"Row count {len(conformed)} outside expected range [{COUNT_LO}, {COUNT_HI}]. "
    "Source sheet may have been modified — review before adjusting bounds."
)
assert all(r["username"] == USERNAME_FILTER for r in conformed), (
    "Filter breach: non-target username present after filter"
)
assert all(r["conversation_id"] for r in conformed), (
    "Empty conversation_id after filter"
)


# COMMAND ----------

# MAGIC %md ## 3. Pre-MERGE deduplication on row_hash

# COMMAND ----------

# Deduplicate source rows on row_hash before MERGE.
# Without this, duplicate source rows with the same hash would both be
# inserted (MERGE matches against the target, not within the source).
seen: dict[str, bool] = {}
deduped: list[dict] = []
for r in conformed:
    if r["row_hash"] not in seen:
        seen[r["row_hash"]] = True
        deduped.append(r)

n_dupes = len(conformed) - len(deduped)
if n_dupes:
    print(f"WARNING: {n_dupes} duplicate row_hashes removed from source before MERGE")
conformed = deduped

# Assert uniqueness after dedup (should always hold)
hashes = [r["row_hash"] for r in conformed]
assert len(set(hashes)) == len(hashes), "Logic error: duplicates remain after dedup"
print(f"Post-dedup source rows   : {len(conformed)} (unique hashes)")


# COMMAND ----------

# MAGIC %md ## 4. Build Spark DataFrame + MERGE

# COMMAND ----------

SCHEMA_DEF = StructType([
    StructField("username",        StringType(),    False),
    StructField("query_text",      StringType(),    True),
    StructField("response_text",   StringType(),    True),
    StructField("chat_step",       LongType(),      False),
    StructField("conversation_id", StringType(),    False),
    StructField("source_tool",     StringType(),    False),
    StructField("event_datetime",  TimestampType(), False),
    StructField("ingested_at",     TimestampType(), False),
    StructField("source_name",     StringType(),    False),
    StructField("row_hash",        StringType(),    False),
])

df_new = spark.createDataFrame(
    [(
        r["username"], r["query_text"], r["response_text"],
        r["chat_step"], r["conversation_id"], r["source_tool"],
        r["event_datetime"], r["ingested_at"], r["source_name"], r["row_hash"]
    ) for r in conformed],
    schema=SCHEMA_DEF,
)
df_new.createOrReplaceTempView("new_rows")
print(f"Source DataFrame rows    : {df_new.count()}")

# Ensure schema + target table exist (SPEC §3.1) BEFORE the MERGE. This notebook
# must be self-sufficient: it cannot rely on _setup_uc_objects having run in a
# separate (serverless) session, and it must never "succeed" by MERGE-ing into a
# table that is absent. CREATE ... IF NOT EXISTS is idempotent and byte-for-byte
# matches the SPEC §3.1 contract in _setup_uc_objects.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
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
assert spark.catalog.tableExists(TABLE), (
    f"Target table {TABLE} does not exist after CREATE TABLE IF NOT EXISTS"
)
print(f"Target table ready       : {TABLE}")

# Idempotent MERGE — insert only rows whose row_hash is not already in target
spark.sql(f"""
MERGE INTO {TABLE} AS tgt
USING new_rows AS src
ON tgt.row_hash = src.row_hash
WHEN NOT MATCHED THEN INSERT *
""")
print("MERGE complete")


# COMMAND ----------

# MAGIC %md ## 5. Post-MERGE verification

# COMMAND ----------

count_total = spark.sql(f"SELECT COUNT(*) AS n FROM {TABLE}").first().n
count_user  = spark.sql(
    f"SELECT COUNT(*) AS n FROM {TABLE} WHERE username = '{USERNAME_FILTER}'"
).first().n
count_other = spark.sql(
    f"SELECT COUNT(*) AS n FROM {TABLE} WHERE username != '{USERNAME_FILTER}'"
).first().n

print(f"Rows in {TABLE}:")
print(f"  total                  : {count_total}")
print(f"  username='{USERNAME_FILTER}' : {count_user}")
print(f"  other                  : {count_other}")

assert count_other == 0, (
    f"Contamination: {count_other} rows with username != '{USERNAME_FILTER}'"
)

# Fail hard if the MERGE did not actually populate the table. count_other==0 and
# hash-uniqueness are both vacuously true for an EMPTY table, so those checks
# alone let an empty/absent-data run pass — assert the table is non-empty, holds
# the expected ~247 rows, and contains ONLY the target user.
assert count_total > 0, (
    f"Target table {TABLE} is empty after MERGE — no rows were ingested"
)
assert COUNT_LO <= count_total <= COUNT_HI, (
    f"Post-MERGE row count {count_total} outside expected range "
    f"[{COUNT_LO}, {COUNT_HI}] — ingestion did not land the expected ~247 rows"
)
assert count_user == count_total, (
    f"Target table {TABLE} holds non-'{USERNAME_FILTER}' rows: "
    f"total={count_total} user={count_user}"
)

# Assert post-MERGE row_hash uniqueness (MERGE must not produce duplicates)
n_hash_dupes = spark.sql(f"""
    SELECT COUNT(*) AS n FROM (
        SELECT row_hash FROM {TABLE}
        GROUP BY row_hash HAVING COUNT(*) > 1
    )
""").first().n
assert n_hash_dupes == 0, (
    f"Post-MERGE: {n_hash_dupes} duplicate row_hashes in {TABLE}"
)
print("row_hash uniqueness      : PASS")


# COMMAND ----------

# MAGIC %md ## 6. Sample row

# COMMAND ----------

sample = spark.sql(f"""
SELECT username, conversation_id, chat_step, source_tool,
       event_datetime, source_name,
       LEFT(query_text, 200)    AS query_preview,
       LEFT(response_text, 200) AS response_preview,
       row_hash
FROM {TABLE}
ORDER BY event_datetime ASC
LIMIT 1
""").first()

if sample is not None:
    print("\n=== SAMPLE ROW (earliest by event_datetime) ===")
    for field in sample.__fields__:
        print(f"  {field:20s}: {getattr(sample, field)}")
else:
    print("WARNING: table is empty — no sample to display")


# COMMAND ----------

# MAGIC %md ## 7. Date distribution

# COMMAND ----------

spark.sql(f"""
SELECT DATE_TRUNC('month', event_datetime) AS month,
       COUNT(*)                            AS rows,
       COUNT(DISTINCT conversation_id)     AS conversations
FROM {TABLE}
GROUP BY 1
ORDER BY 1
""").show(20, truncate=False)



# COMMAND ----------

# Sentinel: must be the last thing that executes.
# The runner validates this output; SUCCESS without it means the notebook
# did not execute its full body.
import json as _j
dbutils.notebook.exit(_j.dumps({
    "sentinel":   "ingest:OK",
    "rows_total": count_total,
    "rows_user":  count_user,
}))
