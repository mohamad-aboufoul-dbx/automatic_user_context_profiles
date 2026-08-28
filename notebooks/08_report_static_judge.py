# Databricks notebook source
import pandas as pd

# COMMAND ----------
DRY_RUN = True
RESULTS_TABLE = "ai_fde_hackathon_catalog.automatic_user_context_profiles.eval_results_static_judge"

try:
    dbutils.widgets.text("DRY_RUN", "true")
    dbutils.widgets.text("RESULTS_TABLE", RESULTS_TABLE)
    DRY_RUN = dbutils.widgets.get("DRY_RUN").strip().lower() == "true"
    RESULTS_TABLE = dbutils.widgets.get("RESULTS_TABLE").strip()
except (NameError, AttributeError):
    pass

# COMMAND ----------
if not DRY_RUN:
    results = spark.table(RESULTS_TABLE).toPandas()
else:
    results = pd.DataFrame(
        [
            {"Username": "SYNTHETIC_USER", "q_key": 1, "Profile_used": "N/a", "data_combo": None, "profile_level": None, "New_or_old_question": "New", "coverage_score": 0.40, "failure_mode": "SYNTHETIC_MISSING_DETAIL", "rationale": "SYNTHETIC", "parse_error": None, "judge_endpoint": "SYNTHETIC", "scored_at": pd.Timestamp("2026-08-28")},
            {"Username": "SYNTHETIC_USER", "q_key": 2, "Profile_used": "N/a", "data_combo": None, "profile_level": None, "New_or_old_question": "Old", "coverage_score": 0.55, "failure_mode": "SYNTHETIC_NONE", "rationale": "SYNTHETIC", "parse_error": None, "judge_endpoint": "SYNTHETIC", "scored_at": pd.Timestamp("2026-08-28")},
            {"Username": "SYNTHETIC_USER", "q_key": 3, "Profile_used": "N/a", "data_combo": None, "profile_level": None, "New_or_old_question": "New", "coverage_score": None, "failure_mode": "SYNTHETIC_UNSCORED", "rationale": "SYNTHETIC", "parse_error": "SYNTHETIC_PARSE_ERROR", "judge_endpoint": "SYNTHETIC", "scored_at": pd.Timestamp("2026-08-28")},
            {"Username": "SYNTHETIC_USER", "q_key": 1, "Profile_used": "Combo_B_Level_2", "data_combo": "B", "profile_level": "level_2_detailed", "New_or_old_question": "New", "coverage_score": 0.70, "failure_mode": "SYNTHETIC_NONE", "rationale": "SYNTHETIC", "parse_error": None, "judge_endpoint": "SYNTHETIC", "scored_at": pd.Timestamp("2026-08-28")},
            {"Username": "SYNTHETIC_USER", "q_key": 2, "Profile_used": "Combo_B_Level_2", "data_combo": "B", "profile_level": "level_2_detailed", "New_or_old_question": "Old", "coverage_score": 0.75, "failure_mode": "SYNTHETIC_MINOR_GAP", "rationale": "SYNTHETIC", "parse_error": None, "judge_endpoint": "SYNTHETIC", "scored_at": pd.Timestamp("2026-08-28")},
            {"Username": "SYNTHETIC_USER", "q_key": 1, "Profile_used": "Combo_C_Level_3", "data_combo": "C", "profile_level": "level_3_evidence", "New_or_old_question": "New", "coverage_score": 0.85, "failure_mode": "SYNTHETIC_NONE", "rationale": "SYNTHETIC", "parse_error": None, "judge_endpoint": "SYNTHETIC", "scored_at": pd.Timestamp("2026-08-28")},
            {"Username": "SYNTHETIC_USER", "q_key": 2, "Profile_used": "Combo_C_Level_3", "data_combo": "C", "profile_level": "level_3_evidence", "New_or_old_question": "Old", "coverage_score": 0.80, "failure_mode": "SYNTHETIC_NONE", "rationale": "SYNTHETIC", "parse_error": None, "judge_endpoint": "SYNTHETIC", "scored_at": pd.Timestamp("2026-08-28")},
        ]
    )

# COMMAND ----------
results["coverage_score"] = pd.to_numeric(results["coverage_score"], errors="coerce")

def print_table(title, frame):
    print(f"\n{title}")
    print("(no rows)" if frame.empty else frame.to_string(index=False))

profile_summary = (
    results.groupby("Profile_used", dropna=False)["coverage_score"]
    .agg(mean_coverage="mean", n_rows="size", n_scored="count")
    .reset_index()
    .sort_values("Profile_used")
)
print_table("(a) Coverage by Profile_used", profile_summary)

combo_summary = (
    results.groupby(["data_combo", "profile_level"], dropna=False)["coverage_score"]
    .agg(mean_coverage="mean", n_rows="size", n_scored="count")
    .reset_index()
    .sort_values(["data_combo", "profile_level"], na_position="first")
)
print_table("(b) Coverage by data_combo and profile_level", combo_summary)

question_summary = (
    results.groupby(["New_or_old_question", "Profile_used"], dropna=False)["coverage_score"]
    .agg(mean_coverage="mean", n_rows="size", n_scored="count")
    .reset_index()
    .sort_values(["New_or_old_question", "Profile_used"])
)
print_table("(c) Coverage by question type and Profile_used", question_summary)

print("\n(d) DELTA vs baseline")
baseline = profile_summary.loc[profile_summary["Profile_used"] == "N/a"]
baseline_mean = baseline.iloc[0]["mean_coverage"] if not baseline.empty and baseline.iloc[0]["n_scored"] else None
for row in profile_summary.itertuples(index=False):
    if row.Profile_used == "N/a":
        continue
    delta = None if baseline_mean is None or row.n_scored == 0 else row.mean_coverage - baseline_mean
    print(f"{row.Profile_used}: {'n/a' if delta is None else f'{delta:+.3f}'}")

failure_counts = (
    results.groupby(["Profile_used", "failure_mode"], dropna=False)
    .size()
    .reset_index(name="count")
    .sort_values(["Profile_used", "count", "failure_mode"], ascending=[True, False, True])
)
print_table("(e) Failure mode counts by Profile_used", failure_counts)

n_rows = len(results)
n_conditions = results["Profile_used"].nunique(dropna=False)
print(f"\nCAVEAT: n is tiny ({n_rows} rows across {n_conditions} conditions, zero replicates), so differences are not statistically reliable.")
