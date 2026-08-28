"""Rebuild all 12 runtime profile files from the UC checkpoint table.

Run this from a Databricks notebook cell after the profile_builder pipeline
completes. It pulls the latest profiles and prepends the agent instruction prefix.

Output: 12 YAML files per user (4 levels x 3 combos), each ready to paste into
an agent's system prompt or custom instructions.

Filename format: {user_slug}_combo_{a|b|c}_{level}.yaml
"""

import os
import re

CATALOG = "ai_fde_hackathon_catalog"
SCHEMA = "automatic_user_context_profiles"
CHECKPOINT_TABLE = f"{CATALOG}.{SCHEMA}.profile_checkpoints_v2"

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else \
    "/Workspace/Users/mohamad.aboufoul@databricks.com/automatic_user_context_profiles/profiles"
PREFIX_PATH = os.path.join(SCRIPT_DIR, "prefix.yaml")
OUTPUT_DIR = SCRIPT_DIR

# --- Agent instruction prefix (loaded from prefix.yaml) ---
# This is prepended to every profile file so agents know how to interpret
# capability levels, confidence scores, interaction patterns, and failure modes.
# Edit prefix.yaml to change the instructions all profiles receive.


def slugify_user(name: str) -> str:
    """Convert 'Mohamad Aboufoul' -> 'mohamad_aboufoul' for filenames."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def build():
    """Pull profiles from UC, prepend prefix, write per-user YAML files."""

    # Load prefix
    if not os.path.exists(PREFIX_PATH):
        print(f"ERROR: Prefix file not found at {PREFIX_PATH}")
        return
    with open(PREFIX_PATH, "r") as f:
        prefix = f.read()
    print(f"Loaded prefix ({len(prefix):,} chars) from {PREFIX_PATH}")

    # Load all profiles from UC
    rows = spark.sql(f"""
        SELECT user_name, data_combo, data_combo_label, profile_level,
               profile_yaml, model_used, generated_at
        FROM {CHECKPOINT_TABLE}
        ORDER BY user_name, data_combo, profile_level
    """).collect()

    if not rows:
        print("ERROR: No profiles found in checkpoint table. Run profile_builder first.")
        return

    # Group by user (supports multiple users in the same table)
    users = set(row['user_name'] for row in rows)
    print(f"Found profiles for {len(users)} user(s): {', '.join(users)}")
    print()

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

    print(f"Built {len(written)} profile files:\n")
    for fname, size in written:
        print(f"  {fname}: {size:,} chars")


build()
