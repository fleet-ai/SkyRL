#!/usr/bin/env python3
"""Export wallst tasks from Supabase to JSON file for SkyRL training.

Queries the eval_tasks and eval_task_versions tables for env_key=wallst tasks
and outputs them in the Fleet task JSON format expected by prepare_dataset.py.

Wallst has ~3,438 tasks across 13 data keys (tmt, skynet, airlines, chemicals,
steel, reits, etc.) with 53 MCP tools (mostly Excel operations).

Usage:
    # Export all wallst tasks
    python -m scripts.export_wallst_tasks --output ~/data/fleet/wallst_tasks.json

    # Export and upload to S3
    python -m scripts.export_wallst_tasks \
        --output ~/data/fleet/wallst_tasks.json \
        --upload-s3 s3://fleet-internal-datasets/v0.4-wallst/openenv/all_tool_use.json

    # Dry run (preview without writing)
    python -m scripts.export_wallst_tasks --output /dev/null --dry-run

Requires:
    SUPABASE_URL and SUPABASE_KEY environment variables to be set.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional


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


def fetch_wallst_tasks(supabase, env_key: str = "wallst") -> List[Dict[str, Any]]:
    """Fetch wallst tasks from eval_tasks and eval_task_versions tables.

    Returns list of task dicts in Fleet task JSON format.
    """
    print(f"Querying eval_tasks for env_key={env_key}...")

    # Fetch all eval_tasks for wallst
    # Supabase client paginates at 1000 by default, so we need to paginate
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

    print(f"Fetched {len(all_tasks)} eval_tasks")

    if not all_tasks:
        print("WARNING: No tasks found. Check env_key and Supabase credentials.")
        return []

    # Fetch task versions for prompts and verifier IDs
    task_ids = [t["id"] for t in all_tasks]
    print(f"Fetching eval_task_versions for {len(task_ids)} tasks...")

    versions_by_task: Dict[str, Dict[str, Any]] = {}
    # Batch the version queries to avoid URL length limits
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
            # Keep latest version per task
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

        # Build the task key from the task record
        task_key = task.get("key") or task.get("task_key") or f"wallst_{task_id}"

        # Extract data key info
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

    # Print per-data-key breakdown
    print(f"\n=== Task Breakdown by Data Key ===")
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
    parser = argparse.ArgumentParser(
        description="Export wallst tasks from Supabase for SkyRL training"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--env-key",
        default="wallst",
        help="Environment key to export (default: wallst)",
    )
    parser.add_argument(
        "--upload-s3",
        default=None,
        help="S3 path to upload the output file (e.g., s3://fleet-internal-datasets/v0.4-wallst/openenv/all_tool_use.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and print stats without writing output",
    )
    args = parser.parse_args()

    supabase = get_supabase_client()
    tasks = fetch_wallst_tasks(supabase, env_key=args.env_key)

    if not tasks:
        print("No tasks to export. Exiting.")
        sys.exit(1)

    if args.dry_run:
        print(f"\nDry run: would write {len(tasks)} tasks to {args.output}")
        # Print a sample task
        print("\nSample task:")
        sample = {k: v for k, v in tasks[0].items() if k != "prompt"}
        sample["prompt"] = tasks[0]["prompt"][:200] + "..."
        print(json.dumps(sample, indent=2))
        return

    # Write output
    output_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    output = {"tasks": tasks}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported {len(tasks)} tasks to {output_path}")

    # Upload to S3 if requested
    if args.upload_s3:
        upload_to_s3(output_path, args.upload_s3)


if __name__ == "__main__":
    main()
