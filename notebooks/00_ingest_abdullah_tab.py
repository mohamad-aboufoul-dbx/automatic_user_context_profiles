# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Ingest Abdullah Tab → raw_conversations_abdullah_said
# MAGIC
# MAGIC Loads the 'Abdullah' tab from the 'Chatbot conversations' Google Sheet
# MAGIC (spreadsheet_id=1NCFFYeCPs-TS3YA6r3Y9y8p0LtUy2cY0-qa8CMl1iEI, gid=1354471250)
# MAGIC into UC table raw_conversations_abdullah_said via idempotent MERGE on row_hash.
# MAGIC Falls back to a local CSV when the Sheets API is unreachable.

# COMMAND ----------

import hashlib
import os
from datetime import datetime, timezone
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, TimestampType
)

CATALOG      = "ai_fde_hackathon_catalog"
SCHEMA       = "automatic_user_context_profiles"
TABLE        = f"{CATALOG}.{SCHEMA}.raw_conversations_abdullah_said"
SOURCE_NAME  = "chatbot_conversations_abdullah"
SPREADSHEET  = "1NCFFYeCPs-TS3YA6r3Y9y8p0LtUy2cY0-qa8CMl1iEI"
RANGE_NAME   = "Abdullah"
# CSV fallback — place raw_abdullah_tab.csv here if Sheets API is unreachable
CSV_FALLBACK = "/Volumes/ai_fde_hackathon_catalog/automatic_user_context_profiles/raw/raw_abdullah_tab.csv"
CSV_LOCAL    = "/dbfs/tmp/raw_abdullah_tab.csv"   # for dbutils.fs.cp from local
USERNAME_FILTER = "abdullah.said"

# COMMAND ----------
# MAGIC %md ## 1. Fetch raw rows (Sheets API → CSV fallback)

def fetch_from_sheets() -> list[dict]:
    """Returns list of raw dicts from the Google Sheets API."""
    import urllib.request, urllib.error, json
    # Use google-auth ADC token if available, else try instance metadata
    try:
        import subprocess
        token = subprocess.check_output(
            ["gcloud", "auth", "application-default", "print-access-token"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception as e:
        raise RuntimeError(f"Could not get gcloud token: {e}")

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

    header = rows_raw[0]
    records = []
    for row in rows_raw[1:]:
        padded = row + [""] * (len(header) - len(row))
        records.append(dict(zip(header, padded[:len(header)])))
    return records


def fetch_from_csv(path: str) -> list[dict]:
    """Returns list of raw dicts from a CSV file."""
    import csv
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(dict(row))
    return records


# Try Sheets API first, fall back to CSV
records = None
source_path = "sheets_api"
errors = []

try:
    records = fetch_from_sheets()
    print(f"Fetched {len(records)} rows from Sheets API")
except Exception as e:
    errors.append(f"Sheets API: {e}")
    print(f"Sheets API unavailable: {e}")

if records is None:
    for fallback in [CSV_FALLBACK, CSV_LOCAL]:
        try:
            records = fetch_from_csv(fallback)
            source_path = fallback
            print(f"Loaded {len(records)} rows from CSV fallback: {fallback}")
            break
        except Exception as e:
            errors.append(f"CSV {fallback}: {e}")

if records is None:
    raise RuntimeError(
        "Both Sheets API and CSV fallback failed:\n" + "\n".join(errors)
    )

print(f"Source: {source_path}")
print(f"Raw row count: {len(records)}")

# COMMAND ----------
# MAGIC %md ## 2. Conform + hash + filter

DATETIME_FMTS = [
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
]


def parse_dt(s: str):
    for fmt in DATETIME_FMTS:
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def row_hash(username, query_text, response_text, chat_step, conversation_id,
             source_tool, event_datetime) -> str:
    """SHA-256 of normalized canonical fields."""
    canonical = "\x1f".join([
        str(username or "").strip().lower(),
        str(query_text or "").strip(),
        str(response_text or "").strip(),
        str(chat_step or "").strip(),
        str(conversation_id or "").strip(),
        str(source_tool or "").strip(),
        str(event_datetime or ""),
    ])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


conformed = []
skipped = 0
ingested_at = datetime.now(timezone.utc)

for rec in records:
    username = (rec.get("Username") or "").strip()

    # Filter to target user only
    if username != USERNAME_FILTER:
        skipped += 1
        continue

    query_text    = rec.get("Query", "") or ""
    response_text = rec.get("Response", "") or ""
    chat_step_raw = rec.get("Chat_step", "0") or "0"
    conversation_id = (rec.get("Conversation_id") or "").strip()
    source_tool   = (rec.get("Tool") or "").strip()
    dt_str        = (rec.get("Datetime") or "").strip()

    try:
        chat_step_int = int(float(chat_step_raw))
    except (ValueError, TypeError):
        chat_step_int = 0

    event_dt = parse_dt(dt_str)
    if event_dt is None:
        print(f"  WARNING: unparseable datetime {dt_str!r} for conv {conversation_id}")
        continue

    h = row_hash(username, query_text, response_text,
                 chat_step_int, conversation_id, source_tool, event_dt.isoformat())

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

print(f"Conformed rows:  {len(conformed)}")
print(f"Skipped (non-{USERNAME_FILTER}): {skipped}")
assert all(r["username"] == USERNAME_FILTER for r in conformed), "Filter breach!"

# COMMAND ----------
# MAGIC %md ## 3. Create Spark DataFrame + idempotent MERGE

from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, TimestampType
)

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
    schema=SCHEMA_DEF
)
df_new.createOrReplaceTempView("new_rows")

print(f"DataFrame rows: {df_new.count()}")

# COMMAND ----------

# Idempotent MERGE — insert only rows whose row_hash is not already present
spark.sql(f"""
MERGE INTO {TABLE} AS tgt
USING new_rows AS src
ON tgt.row_hash = src.row_hash
WHEN NOT MATCHED THEN INSERT *
""")
print("MERGE complete")

# COMMAND ----------
# MAGIC %md ## 4. Verify

count_total = spark.sql(f"SELECT COUNT(*) AS n FROM {TABLE}").first().n
count_user  = spark.sql(
    f"SELECT COUNT(*) AS n FROM {TABLE} WHERE username = '{USERNAME_FILTER}'"
).first().n
count_other = spark.sql(
    f"SELECT COUNT(*) AS n FROM {TABLE} WHERE username != '{USERNAME_FILTER}'"
).first().n

print(f"Total rows in {TABLE}: {count_total}")
print(f"  username='{USERNAME_FILTER}': {count_user}")
print(f"  other usernames:             {count_other}")
assert count_other == 0, f"Contamination: {count_other} non-{USERNAME_FILTER} rows!"

# COMMAND ----------

# Sample row for human review
print("\n=== SAMPLE ROW ===")
sample = spark.sql(f"""
SELECT username, conversation_id, chat_step, source_tool,
       event_datetime, source_name,
       LEFT(query_text, 200) AS query_preview,
       LEFT(response_text, 200) AS response_preview,
       row_hash
FROM {TABLE}
ORDER BY event_datetime ASC
LIMIT 1
""").first()
for field in sample.__fields__:
    print(f"  {field:20s}: {getattr(sample, field)}")

# COMMAND ----------
# MAGIC %md ## 5. Date distribution

spark.sql(f"""
SELECT DATE_TRUNC('month', event_datetime) AS month,
       COUNT(*) AS rows,
       COUNT(DISTINCT conversation_id) AS conversations
FROM {TABLE}
GROUP BY 1
ORDER BY 1
""").show(20, truncate=False)
