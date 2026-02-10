#!/usr/bin/env python3
"""Export tasks for one or more env_keys from Supabase to JSON file for SkyRL training.

Queries the eval_tasks and eval_task_versions tables and outputs them in the
Fleet task JSON format expected by prepare_dataset.py.

Usage:
    # Export jira + outlook tasks
    python -m scripts.export_env_tasks \
        --env-keys jira-product-discovery jira-service-management outlook \
        --output ~/data/fleet/jira_outlook_tasks.json

    # Export and upload to S3
    python -m scripts.export_env_tasks \
        --env-keys jira-product-discovery jira-service-management outlook \
        --output ~/data/fleet/jira_outlook_tasks.json \
        --upload-s3 s3://fleet-internal-datasets/v0.4-jira-outlook/openenv/all_tool_use.json

    # Dry run (preview without writing)
    python -m scripts.export_env_tasks \
        --env-keys jira-product-discovery jira-service-management outlook \
        --output /dev/null --dry-run

Requires:
    SUPABASE_URL and SUPABASE_KEY environment variables to be set.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List


def get_supabase_client():
    """Create Supabase client from environment variables."""
    try:
        from supabase import create_client
    except ImportError:
        print("ERROR: supabase-py not installed. Install with: pip install supabase")
        sys.exit(1)

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        sys.exit(1)

    return create_client(url, key)


def fetch_tasks_for_env(supabase, env_key: str) -> List[Dict[str, Any]]:
    """Fetch tasks for a single env_key from eval_tasks and eval_task_versions.

    Returns list of task dicts in Fleet task JSON format.
    """
    print(f"\nQuerying eval_tasks for env_key={env_key}...")

    all_tasks = []
    page_size = 1000
    offset = 0

    while True:
        response = (
            supabase.table("eval_tasks")
            .select("*")
            .eq("env_key", env_key)
            .range(offset, offset + page_size - 1)
            .execute()
        )

        batch = response.data
        if not batch:
            break

        all_tasks.extend(batch)
        print(f"  Fetched {len(all_tasks)} tasks so far...")

        if len(batch) < page_size:
            break
        offset += page_size

    print(f"Fetched {len(all_tasks)} eval_tasks for {env_key}")

    if not all_tasks:
        print(f"WARNING: No tasks found for env_key={env_key}")
        return []

    # Fetch task versions for prompts and verifier IDs
    task_ids = [t["id"] for t in all_tasks]
    print(f"Fetching eval_task_versions for {len(task_ids)} tasks...")

    versions_by_task: Dict[str, Dict[str, Any]] = {}
    batch_size = 100
    for i in range(0, len(task_ids), batch_size):
        batch_ids = task_ids[i : i + batch_size]
        response = (
            supabase.table("eval_task_versions")
            .select("*")
            .in_("task_id", batch_ids)
            .order("version_no", desc=True)
            .execute()
        )

        for version in response.data:
            tid = version["task_id"]
            if tid not in versions_by_task:
                versions_by_task[tid] = version

        if (i + batch_size) % 500 == 0 or i + batch_size >= len(task_ids):
            print(f"  Fetched versions for {min(i + batch_size, len(task_ids))}/{len(task_ids)} tasks...")

    print(f"Fetched versions for {len(versions_by_task)} tasks")

    # Build task dicts in Fleet format
    task_dicts = []
    skipped = 0
    data_key_counts: Dict[str, int] = defaultdict(int)

    for task in all_tasks:
        task_id = task["id"]
        version = versions_by_task.get(task_id)

        if not version:
            skipped += 1
            continue

        prompt = version.get("prompt") or task.get("prompt")
        if not prompt:
            skipped += 1
            continue

        task_key = task.get("key") or task.get("task_key") or f"{env_key}_{task_id}"
        env_data_key = task.get("env_data_key") or task.get("data_key")
        env_data_version = task.get("env_data_version") or task.get("data_version")
        env_version = task.get("env_version") or task.get("version")
        verifier_id = version.get("verifier_id") or task.get("verifier_id")

        task_dict = {
            "key": task_key,
            "prompt": prompt,
            "env_key": env_key,
            "env_id": env_key,
            "env_version": env_version,
            "env_data_key": env_data_key,
            "env_data_version": env_data_version,
            "task_modality": "tool_use",
            "verifier_id": verifier_id,
            "env_variables": task.get("env_variables") or {},
        }

        task_dicts.append(task_dict)
        data_key_counts[env_data_key or "unknown"] += 1

    if skipped > 0:
        print(f"WARNING: Skipped {skipped} tasks (missing version or prompt)")

    print(f"\n=== {env_key} Task Breakdown by Data Key ===")
    for dk in sorted(data_key_counts.keys()):
        print(f"  {dk}: {data_key_counts[dk]} tasks")
    print(f"  TOTAL: {len(task_dicts)} tasks")

    return task_dicts


def upload_to_s3(local_path: str, s3_path: str):
    """Upload file to S3 using aws CLI."""
    print(f"\nUploading to {s3_path}...")
    result = subprocess.run(
        ["aws", "s3", "cp", local_path, s3_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: S3 upload failed: {result.stderr}")
        sys.exit(1)
    print(f"Uploaded successfully to {s3_path}")


def main():
    parser = argparse.ArgumentParser(description="Export tasks from Supabase for SkyRL training")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--env-keys",
        nargs="+",
        required=True,
        help="One or more environment keys to export (e.g., jira-product-discovery outlook)",
    )
    parser.add_argument(
        "--upload-s3",
        default=None,
        help="S3 path to upload the output file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and print stats without writing output",
    )
    args = parser.parse_args()

    supabase = get_supabase_client()

    all_tasks = []
    for env_key in args.env_keys:
        tasks = fetch_tasks_for_env(supabase, env_key=env_key)
        all_tasks.extend(tasks)

    if not all_tasks:
        print("No tasks to export. Exiting.")
        sys.exit(1)

    # Print combined summary
    env_counts = defaultdict(int)
    for t in all_tasks:
        env_counts[t["env_key"]] += 1
    print(f"\n=== Combined Export Summary ===")
    for ek in sorted(env_counts.keys()):
        print(f"  {ek}: {env_counts[ek]} tasks")
    print(f"  TOTAL: {len(all_tasks)} tasks")

    if args.dry_run:
        print(f"\nDry run: would write {len(all_tasks)} tasks to {args.output}")
        return

    # Write output
    output_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    output = {"tasks": all_tasks}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported {len(all_tasks)} tasks to {output_path}")

    if args.upload_s3:
        upload_to_s3(output_path, args.upload_s3)


if __name__ == "__main__":
    main()
