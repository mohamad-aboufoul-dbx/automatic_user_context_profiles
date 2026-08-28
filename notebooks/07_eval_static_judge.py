# Databricks notebook source
# Static judge eval: score PRE-CAPTURED chatbot answers with a holistic coverage rubric.
# Compares baseline (no profile) vs user-context-profile injection. No live agent calls.
import json
import datetime
import pandas as pd


def _s(x):
    return "" if pd.isna(x) else str(x)

_DEFAULTS = {
    "DRY_RUN": "true",
    "INPUT_TABLE": "ai_fde_hackathon_catalog.automatic_user_context_profiles.chatbot_evaluations_mohamad",
    "OUTPUT_TABLE": "ai_fde_hackathon_catalog.automatic_user_context_profiles.eval_results_static_judge",
    "JUDGE_ENDPOINT": "databricks-claude-sonnet-4",
    "ROW_LIMIT": "0",
}
try:
    for _k, _v in _DEFAULTS.items():
        dbutils.widgets.text(_k, _v)
    _W = {_k: dbutils.widgets.get(_k) for _k in _DEFAULTS}
except Exception:
    _W = dict(_DEFAULTS)

DRY_RUN = str(_W["DRY_RUN"]).strip().lower() in ("true", "1", "yes", "y")
INPUT_TABLE = _W["INPUT_TABLE"]
OUTPUT_TABLE = _W["OUTPUT_TABLE"]
JUDGE_ENDPOINT = _W["JUDGE_ENDPOINT"]
ROW_LIMIT = int(str(_W["ROW_LIMIT"] or "0").strip())
print(f"DRY_RUN={DRY_RUN} judge={JUDGE_ENDPOINT} in={INPUT_TABLE} out={OUTPUT_TABLE} limit={ROW_LIMIT}")

# COMMAND ----------

SYSTEM_PROMPT = (
    "You are a strict evaluator. Grade how well a chatbot RESPONSE satisfies the EXPECTED content "
    "for a QUESTION. Judge only coverage of expected content, not style/length/format. coverage_score "
    "rubric: 1.0 = fully covers essentially all expected content with correct specifics; 0.75 = covers "
    "most, minor gaps; 0.5 = covers about half, notable gaps; 0.25 = touches the topic but misses most; "
    "0.0 = off-topic/wrong/does not address it. Also return failure_mode, one of: covers_well, partial, "
    "too_generic, off_topic, wrong_specifics. Return ONLY a single JSON object, no prose, no code fences: "
    '{"coverage_score": <float>, "failure_mode": "<one of the five>", "rationale": "<=2 sentences"}'
)
VALID_MODES = {"covers_well", "partial", "too_generic", "off_topic", "wrong_specifics"}
_RUBRIC = (0.0, 0.25, 0.5, 0.75, 1.0)


def decode_profile(profile_used):
    p = (profile_used or "").strip()
    combo = "B" if p.startswith("Combo_B") else ("C" if p.startswith("Combo_C") else None)
    level = "level_2_detailed" if p.endswith("Level_2") else ("level_3_evidence" if p.endswith("Level_3") else None)
    return combo, level


def parse_judge_json(text):
    """Strip code fences, extract the first balanced {...} ignoring braces inside strings, json.loads."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    s = s.strip()
    start = s.find("{")
    if start < 0:
        raise ValueError("no JSON object found in judge output")
    depth, in_str, esc, end = 0, False, False, -1
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError("unbalanced JSON object in judge output")
    obj = json.loads(s[start:end])
    score = float(obj["coverage_score"])
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"coverage_score out of range: {score}")
    score = min(_RUBRIC, key=lambda s2: abs(s2 - score))
    mode = str(obj.get("failure_mode", "")).strip()
    if mode not in VALID_MODES:
        raise ValueError(f"invalid failure_mode: {mode!r}")
    return score, mode, str(obj.get("rationale", ""))[:1000]

# COMMAND ----------

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        host = ctx.apiUrl().get()
        token = ctx.apiToken().get()
        _CLIENT = OpenAI(base_url=f"{host}/serving-endpoints", api_key=token)
    return _CLIENT


def judge_row(query, expected, response):
    """Return (coverage_score, failure_mode, rationale, parse_error). Never raises."""
    user_msg = f"QUESTION:\n{query}\n\nEXPECTED:\n{expected}\n\nRESPONSE:\n{response}"
    raw = ""
    try:
        # NOTE: never set response_format -- Databricks Claude endpoints reject it.
        out = _client().chat.completions.create(
            model=JUDGE_ENDPOINT,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=500,
        )
        raw = out.choices[0].message.content or ""
        score, mode, rationale = parse_judge_json(raw)
        return score, mode, rationale, None
    except Exception as e:
        return None, None, f"RAW: {raw[:500]}", str(e)


def stub_judge(query, expected, response):
    """DRY_RUN only: trivial token-overlap stub so aggregation prints without a live endpoint."""
    exp = {t for t in "".join(c if c.isalnum() else " " for c in (expected or "").lower()).split() if len(t) > 3}
    got = set("".join(c if c.isalnum() else " " for c in (response or "").lower()).split())
    frac = (len(exp & got) / len(exp)) if exp else 0.0
    score = min([1.0, 0.75, 0.5, 0.25, 0.0], key=lambda s: abs(s - frac))
    mode = {1.0: "covers_well", 0.75: "partial", 0.5: "partial", 0.25: "too_generic", 0.0: "off_topic"}[score]
    return score, mode, "stubbed dry-run score from token overlap", None

# COMMAND ----------

SAMPLE_ROWS = [
    {"Username": "synthetic_user_1", "Query": "What is the capital of France?",
     "Response": "Paris is the capital of France.", "Tool": "synthetic_tool",
     "New_or_old_question": "New", "Expected_elements_in_response": "Paris; located on the Seine",
     "Profile_used": "N/a"},
    {"Username": "synthetic_user_1", "Query": "What is the capital of France?",
     "Response": "Paris, on the Seine river, is the capital of France.", "Tool": "synthetic_tool",
     "New_or_old_question": "New", "Expected_elements_in_response": "Paris; located on the Seine",
     "Profile_used": "Combo_B_Level_2"},
    {"Username": "synthetic_user_2", "Query": "How many days are in a leap year?",
     "Response": "It depends on the calendar you use.", "Tool": "synthetic_tool",
     "New_or_old_question": "Old", "Expected_elements_in_response": "366 days",
     "Profile_used": "N/a"},
]

if DRY_RUN:
    src = pd.DataFrame(SAMPLE_ROWS)
else:
    sql = f"SELECT * FROM {INPUT_TABLE}" + (f" LIMIT {ROW_LIMIT}" if ROW_LIMIT > 0 else "")
    src = spark.sql(sql).toPandas()
print(f"rows to score: {len(src)}")

# COMMAND ----------

scored_at = datetime.datetime.now()
rows = []
for _, r in src.iterrows():
    try:
        expected = _s(r.get("Expected_elements_in_response"))
        scorer = stub_judge if DRY_RUN else judge_row
        score, mode, rationale, err = scorer(_s(r.get("Query")), expected, _s(r.get("Response")))
        combo, level = decode_profile(_s(r.get("Profile_used")))
        rows.append({
            "Username": _s(r.get("Username")), "q_key": int(len(expected)), "Profile_used": _s(r.get("Profile_used")),
            "data_combo": combo, "profile_level": level, "New_or_old_question": _s(r.get("New_or_old_question")),
            "coverage_score": score, "failure_mode": mode, "rationale": rationale, "parse_error": err,
            "judge_endpoint": JUDGE_ENDPOINT, "scored_at": scored_at,
        })
    except Exception as e:
        rows.append({
            "Username": _s(r.get("Username")), "q_key": None, "Profile_used": _s(r.get("Profile_used")),
            "data_combo": None, "profile_level": None, "New_or_old_question": _s(r.get("New_or_old_question")),
            "coverage_score": None, "failure_mode": None, "rationale": None, "parse_error": f"row_error: {e}",
            "judge_endpoint": JUDGE_ENDPOINT, "scored_at": scored_at,
        })

results = pd.DataFrame(rows, columns=[
    "Username", "q_key", "Profile_used", "data_combo", "profile_level", "New_or_old_question",
    "coverage_score", "failure_mode", "rationale", "parse_error", "judge_endpoint", "scored_at"])
results["coverage_score"] = results["coverage_score"].astype("float64")

# COMMAND ----------

print(results[["Profile_used", "data_combo", "profile_level", "New_or_old_question",
               "coverage_score", "failure_mode"]].to_string(index=False))
print("\nmean coverage_score by Profile_used:")
print(results.groupby("Profile_used", dropna=False)["coverage_score"].mean().to_string())
n_err = int(results["parse_error"].notna().sum())
print(f"\nparse_errors: {n_err}/{len(results)}")

if not DRY_RUN:
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
    schema = StructType([
        StructField("Username", StringType(), True), StructField("q_key", IntegerType(), True),
        StructField("Profile_used", StringType(), True), StructField("data_combo", StringType(), True),
        StructField("profile_level", StringType(), True), StructField("New_or_old_question", StringType(), True),
        StructField("coverage_score", DoubleType(), True), StructField("failure_mode", StringType(), True),
        StructField("rationale", StringType(), True), StructField("parse_error", StringType(), True),
        StructField("judge_endpoint", StringType(), True), StructField("scored_at", TimestampType(), True),
    ])
    out = results.astype(object).where(pd.notna(results), None)
    spark.createDataFrame(out, schema=schema).write.mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE)
    print(f"wrote {len(results)} rows -> {OUTPUT_TABLE}")
else:
    print("DRY_RUN: skipped Spark write")
