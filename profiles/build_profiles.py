"""Rebuild all runtime profile files from the UC checkpoint table.

Run this from a Databricks notebook cell after the profile_builder pipeline
completes. It pulls the latest profiles and prepends the agent instruction prefix.

Output: 12 YAML files per user (4 levels x 3 combos), each ready to paste into
an agent's system prompt or custom instructions.

Filename format: {user_slug}_combo_{a|b|c}_{level}.yaml

The user_slug is derived dynamically from the user_name column in the checkpoint
table (e.g., "Mohamad Aboufoul" -> "mohamad_aboufoul"). Supports multiple users
side-by-side.
"""

import os
import re

# --- Configuration ---
CATALOG = "ai_fde_hackathon_catalog"
SCHEMA = "automatic_user_context_profiles"
CHECKPOINT_TABLE = f"{CATALOG}.{SCHEMA}.profile_checkpoints_v2"

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else \
    "/Workspace/Users/mohamad.aboufoul@databricks.com/automatic_user_context_profiles/profiles"
PREFIX_PATH = os.path.join(SCRIPT_DIR, "prefix.yaml")
OUTPUT_DIR = SCRIPT_DIR


def slugify_user(name: str) -> str:
    """Convert a display name to a filename slug.
    
    Examples:
        'Mohamad Aboufoul' -> 'mohamad_aboufoul'
        'Jane O'Brien' -> 'jane_o_brien'
    """
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def build(user_filter: str = None):
    """Pull profiles from UC, prepend prefix, write per-user YAML files.
    
    Args:
        user_filter: If provided, only build profiles for this user name.
                     If None, builds for ALL users in the checkpoint table.
    """

    # --- Load the agent instruction prefix ---
    # This is prepended to every profile file so agents know how to interpret
    # capability levels, confidence scores, interaction patterns, and failure modes.
    if not os.path.exists(PREFIX_PATH):
        print(f"ERROR: Prefix file not found at {PREFIX_PATH}")
        print(f"  Expected: {PREFIX_PATH}")
        print(f"  The prefix tells agents how to interpret the profile YAML.")
        return
    with open(PREFIX_PATH, "r") as f:
        prefix = f.read()
    print(f"Loaded prefix ({len(prefix):,} chars) from prefix.yaml")

    # --- Load profiles from UC ---
    where_clause = f"WHERE user_name = '{user_filter}'" if user_filter else ""
    rows = spark.sql(f"""
        SELECT user_name, data_combo, data_combo_label, profile_level,
               profile_yaml, model_used, generated_at
        FROM {CHECKPOINT_TABLE}
        {where_clause}
        ORDER BY user_name, data_combo, profile_level
    """).collect()

    if not rows:
        msg = f" for user '{user_filter}'" if user_filter else ""
        print(f"ERROR: No profiles found{msg} in {CHECKPOINT_TABLE}.")
        print(f"  Run the profile_builder notebook first.")
        return

    # Group by user (supports multiple users in the same table)
    users = set(row['user_name'] for row in rows)
    print(f"Found profiles for {len(users)} user(s): {', '.join(sorted(users))}")
    print()

    # --- Write profile files ---
    written = []
    for row in rows:
        user_slug = slugify_user(row['user_name'])
        combo = row['data_combo'].lower()
        level = row['profile_level']
        label = row['data_combo_label']
        model = row['model_used']
        yaml_content = row['profile_yaml']
        generated_at = row['generated_at']

        filename = f"{user_slug}_combo_{combo}_{level}.yaml"
        header = (
            f"# User: {row['user_name']}\n"
            f"# Profile: {level} | Data source: {label}\n"
            f"# Generated: {generated_at}\n"
            f"# Model: {model}\n"
            f"# Source: {CHECKPOINT_TABLE}\n\n"
        )

        full_content = header + prefix + "\n" + yaml_content
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            f.write(full_content)
        written.append((filename, len(full_content)))

    print(f"Built {len(written)} profile files in {OUTPUT_DIR}/:\n")
    for fname, size in written:
        print(f"  {fname}: {size:,} chars")
    print(f"\nEach file contains: header + prefix ({len(prefix):,} chars) + profile YAML")


# --- Run ---
# To build for a specific user only:
#   build(user_filter="Mohamad Aboufoul")
# To build for all users in the checkpoint table:
build()

