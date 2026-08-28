# Databricks notebook source
# Report for the profile-context eval harness.
#
# Reads the eval_runs table written by 05_run_eval and summarizes, per arm:
#   pass_rate                  = mean(answered)
#   mean_iterations_to_answer  = mean(iterations_to_answer) over answered rows
#   n                          = rows in the arm
#
# The headline comparison is the no_context control against each level arm.

# COMMAND ----------

dbutils.widgets.text("CATALOG", "ai_fde_hackathon_catalog", "Catalog")
dbutils.widgets.text("SCHEMA", "automatic_user_context_profiles", "Schema")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
TABLE = f"{CATALOG}.{SCHEMA}.eval_runs"
print(f"reading {TABLE}")

# COMMAND ----------

CONTROL_ARM = "no_context"

per_arm = spark.sql(
    f"""
    SELECT
      arm,
      COUNT(*)                                        AS n,
      AVG(CASE WHEN answered THEN 1.0 ELSE 0.0 END)   AS pass_rate,
      AVG(CASE WHEN answered THEN iterations_to_answer END)
                                                      AS mean_iterations_to_answer
    FROM {TABLE}
    GROUP BY arm
    ORDER BY arm
    """
)
display(per_arm)

# COMMAND ----------

# Control vs each level arm: deltas in pass rate and turns-to-answer.
# A negative iterations delta means the profile arm answered in fewer turns.
comparison = spark.sql(
    f"""
    WITH per_arm AS (
      SELECT
        arm,
        COUNT(*)                                      AS n,
        AVG(CASE WHEN answered THEN 1.0 ELSE 0.0 END) AS pass_rate,
        AVG(CASE WHEN answered THEN iterations_to_answer END)
                                                      AS mean_iterations_to_answer
      FROM {TABLE}
      GROUP BY arm
    ),
    control AS (
      SELECT pass_rate, mean_iterations_to_answer
      FROM per_arm
      WHERE arm = '{CONTROL_ARM}'
    )
    SELECT
      a.arm,
      a.n,
      a.pass_rate,
      a.mean_iterations_to_answer,
      c.pass_rate                                        AS control_pass_rate,
      c.mean_iterations_to_answer                        AS control_mean_iterations,
      a.pass_rate - c.pass_rate                          AS pass_rate_delta,
      a.mean_iterations_to_answer - c.mean_iterations_to_answer
                                                         AS iterations_delta
    FROM per_arm a
    CROSS JOIN control c
    WHERE a.arm <> '{CONTROL_ARM}'
    ORDER BY a.arm
    """
)
display(comparison)
