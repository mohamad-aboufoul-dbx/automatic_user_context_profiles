# Databricks notebook source
# Profile-context eval harness.
#
# Question this measures: does injecting a user-context profile (at increasing
# levels of detail) let the agent satisfy an eval question in FEWER follow-up
# turns than running with no context at all?
#
# Each eval task is run once per arm. An LLM judge decides, after every agent
# turn, whether the answer covers all expected answer elements. The number of
# turns needed is the primary metric (`iterations_to_answer`).
#
# Profile text is loaded at runtime from Unity Catalog -- nothing about any
# profile body is hardcoded in this notebook.

# COMMAND ----------

# --- Widgets ---------------------------------------------------------------
dbutils.widgets.text("MAX_ITERATIONS", "5", "Max agent turns per task")
dbutils.widgets.dropdown("DATA_COMBO", "C", ["A", "B", "C"], "Data combo")
dbutils.widgets.text("JUDGE_ENDPOINT", "databricks-claude-opus-4-8", "Judge endpoint")
dbutils.widgets.text("AGENT_ENDPOINT", "databricks-claude-opus-4-8", "Agent endpoint")
dbutils.widgets.dropdown("DRY_RUN", "true", ["true", "false"], "Dry run (no table write)")
dbutils.widgets.text("CATALOG", "ai_fde_hackathon_catalog", "Catalog")
dbutils.widgets.text("SCHEMA", "automatic_user_context_profiles", "Schema")
dbutils.widgets.text("USER_NAME", "abdullah.said", "Profile user_name")

# COMMAND ----------

# --- Typed config ----------------------------------------------------------
MAX_ITERATIONS = int(dbutils.widgets.get("MAX_ITERATIONS"))
DATA_COMBO = dbutils.widgets.get("DATA_COMBO")
JUDGE_ENDPOINT = dbutils.widgets.get("JUDGE_ENDPOINT")
AGENT_ENDPOINT = dbutils.widgets.get("AGENT_ENDPOINT")
DRY_RUN = dbutils.widgets.get("DRY_RUN").strip().lower() == "true"
CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
USER_NAME = dbutils.widgets.get("USER_NAME")

print(f"data_combo={DATA_COMBO} max_iterations={MAX_ITERATIONS} dry_run={DRY_RUN}")
print(f"target={CATALOG}.{SCHEMA}.eval_runs")

# COMMAND ----------

# --- Arms ------------------------------------------------------------------
# 'no_context' is the WITHOUT-context control: the agent gets the question and
# nothing else. The four level_* arms are the WITH-context conditions, each
# injecting the profile rendered at that level of detail.
ARMS = [
    "no_context",
    "level_0_cold_start",
    "level_1_compact",
    "level_2_detailed",
    "level_3_evidence",
]

# COMMAND ----------

# --- Eval tasks ------------------------------------------------------------
# Tasks live in eval/tasks/T*.json. Each file is one task dict with keys:
#   id, question, expected_answer_elements (list[str])
import glob
import json
import os


def load_tasks(pattern: str = "eval/tasks/T*.json") -> list:
    """Load eval task dicts from disk, falling back to a tiny inline set."""
    tasks = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            tasks.append(json.load(fh))

    if not tasks:
        print(f"No task files matched {pattern!r}; using inline example tasks.")
        tasks = [
            {
                "id": "T000_example_scope",
                "question": "Which datasets should I use for my current project, and why?",
                "expected_answer_elements": [
                    "names the relevant datasets",
                    "explains why each dataset fits the project",
                ],
            },
            {
                "id": "T001_example_preferences",
                "question": "Summarize my reporting preferences and my usual output format.",
                "expected_answer_elements": [
                    "states the preferred output format",
                    "states the preferred level of detail",
                ],
            },
        ]

    print(f"Loaded {len(tasks)} eval task(s): {[t['id'] for t in tasks]}")
    return tasks


TASKS = load_tasks()

# COMMAND ----------

# --- Profile loading (runtime, from Unity Catalog) --------------------------
# Lookup is generic on (user_name, profile_level, data_combo) so any level --
# including level_3_evidence -- resolves the same way. Primary table is
# user_context_profiles; profile_checkpoints_v2 is the fallback.
#
# The returned text is intentionally never printed or logged.

# Arm -> profile_level column value.
ARM_TO_PROFILE_LEVEL = {
    "level_0_cold_start": "level_0_cold_start",
    "level_1_compact": "level_1_compact",
    "level_2_detailed": "level_2_detailed",
    "level_3_evidence": "level_3_evidence",
}

# Widget value -> data_combo column value.
DATA_COMBO_MAP = {"A": "A", "B": "B", "C": "C"}

_PROFILE_QUERY = """
SELECT profile_yaml
FROM {catalog}.{schema}.{table}
WHERE user_name = '{user_name}'
  AND profile_level = '{profile_level}'
  AND data_combo = '{data_combo}'
LIMIT 1
"""


def _query_profile(table: str, user_name: str, profile_level: str, data_combo: str) -> str:
    """Return profile_yaml from one table, or '' when there is no matching row."""
    sql = _PROFILE_QUERY.format(
        catalog=CATALOG,
        schema=SCHEMA,
        table=table,
        user_name=user_name,
        profile_level=profile_level,
        data_combo=data_combo,
    )
    rows = spark.sql(sql).collect()
    if not rows:
        return ""
    return rows[0]["profile_yaml"] or ""


def load_profile_text(arm: str, data_combo: str, user_name: str) -> str:
    """Load the profile body for an arm. Returns '' for the no-context control."""
    if arm == "no_context":
        return ""

    profile_level = ARM_TO_PROFILE_LEVEL[arm]
    combo = DATA_COMBO_MAP[data_combo]

    text = _query_profile("user_context_profiles", user_name, profile_level, combo)
    if not text:
        text = _query_profile("profile_checkpoints_v2", user_name, profile_level, combo)

    # Log only presence and size -- never the body.
    print(f"  profile[{arm}/{combo}]: {'found' if text else 'MISSING'} ({len(text)} chars)")
    return text


# COMMAND ----------

# --- Agent call ------------------------------------------------------------
from databricks.sdk import WorkspaceClient

_workspace = WorkspaceClient()
_openai_client = _workspace.serving_endpoints.get_open_ai_client()


def call_agent(system_or_context: str, messages: list, endpoint: str) -> str:
    """Thin wrapper over a serving endpoint. Returns the assistant text."""
    payload = []
    if system_or_context:
        payload.append({"role": "system", "content": system_or_context})
    payload.extend(messages)

    response = _openai_client.chat.completions.create(
        model=endpoint,
        messages=payload,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


# COMMAND ----------

# --- LLM judge -------------------------------------------------------------
# The judge is an LLM served behind JUDGE_ENDPOINT. It is a strict rubric
# checker, not a data/analytics assistant.

_JUDGE_SYSTEM = (
    "You are a strict evaluation judge. You are given a question, a list of "
    "expected answer elements, and a candidate answer. Decide whether the "
    "candidate answer covers EVERY expected element. Partial or vague coverage "
    "of an element counts as NOT covered.\n"
    'Reply with a single JSON object and nothing else: '
    '{"answered": true|false, "missing": ["<element not covered>", ...]}\n'
    'When every element is covered, "missing" must be an empty list.'
)


def _extract_json(raw: str) -> dict:
    """Parse a JSON object out of a model reply, tolerating code fences."""
    text = raw.strip()
    if text.startswith("```"):
        # Drop the opening fence (with optional language tag) and closing fence.
        text = text.split("\n", 1)[-1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("judge reply contained no JSON object")
    return json.loads(text[start : end + 1])


def judge_answer(question: str, expected_answer_elements: list, candidate_answer: str, endpoint: str) -> dict:
    """Return {'answered': bool, 'missing': list[str]}."""
    elements = "\n".join(f"- {e}" for e in expected_answer_elements)
    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"EXPECTED ANSWER ELEMENTS:\n{elements}\n\n"
        f"CANDIDATE ANSWER:\n{candidate_answer}"
    )

    raw = call_agent(_JUDGE_SYSTEM, [{"role": "user", "content": user_prompt}], endpoint)

    try:
        verdict = _extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        # Unparseable verdict is treated as "not answered" so the loop continues.
        return {"answered": False, "missing": list(expected_answer_elements)}

    return {
        "answered": bool(verdict.get("answered", False)),
        "missing": list(verdict.get("missing") or []),
    }


# COMMAND ----------

# --- Multi-turn loop -------------------------------------------------------
# Sentinel convention: iterations_to_answer is None when the agent never
# satisfied the judge within max_iterations.


def run_task_for_arm(task: dict, arm: str, data_combo: str, max_iterations: int) -> dict:
    """Run one task under one arm, returning a single result row."""
    context = load_profile_text(arm, data_combo, USER_NAME)

    messages = [{"role": "user", "content": task["question"]}]
    answered = False
    iterations_to_answer = None
    final_answer = ""

    for iteration in range(1, max_iterations + 1):
        final_answer = call_agent(context, messages, AGENT_ENDPOINT)
        verdict = judge_answer(
            task["question"],
            task["expected_answer_elements"],
            final_answer,
            JUDGE_ENDPOINT,
        )

        if verdict["answered"]:
            answered = True
            iterations_to_answer = iteration
            break

        # Feed the judge's gaps back as the next follow-up turn.
        missing = "\n".join(f"- {m}" for m in verdict["missing"]) or "- (unspecified gaps)"
        messages.append({"role": "assistant", "content": final_answer})
        messages.append(
            {
                "role": "user",
                "content": "That answer is still missing the following. Please address them:\n" + missing,
            }
        )

    return {
        "task_id": task["id"],
        "arm": arm,
        "data_combo": data_combo,
        "answered": answered,
        "iterations_to_answer": iterations_to_answer,
        "final_answer_len": len(final_answer),
        "max_iterations": max_iterations,
    }


# COMMAND ----------

# --- Driver ----------------------------------------------------------------
results = []
for task in TASKS:
    for arm in ARMS:
        print(f"[{task['id']}] arm={arm}")
        row = run_task_for_arm(task, arm, DATA_COMBO, MAX_ITERATIONS)
        print(f"  answered={row['answered']} iterations={row['iterations_to_answer']}")
        results.append(row)

print(f"\nCollected {len(results)} result row(s).")

# COMMAND ----------

# --- Persist (guarded) -----------------------------------------------------
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.eval_runs"

if DRY_RUN:
    print(f"DRY_RUN=true -- NOT writing to {TARGET_TABLE}.")
    for row in results:
        print(row)
    display(results)
else:
    spark.createDataFrame(results).write.mode("append").saveAsTable(TARGET_TABLE)
    print(f"Appended {len(results)} row(s) to {TARGET_TABLE}.")
