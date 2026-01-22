#!/usr/bin/env python3
"""Debug Fleet version mismatch issue.

Usage:
    FLEET_API_KEY=<key> python scripts/debug_fleet_version.py
"""

import json
import os
import fleet

def main():
    # 1. Load booking tasks from API
    print("=" * 60)
    print("1. Loading booking tasks from Fleet API...")
    print("=" * 60)

    api_tasks = fleet.load_tasks(env_key="booking")
    print(f"Loaded {len(api_tasks)} booking tasks from API")

    if api_tasks:
        from collections import Counter
        versions = Counter(t.version for t in api_tasks)
        print(f"\nVersions available in API:")
        for v, c in versions.most_common():
            print(f"  {v}: {c}")

    # 2. Load sample file
    print("\n" + "=" * 60)
    print("2. Loading sample file...")
    print("=" * 60)

    sample_path = os.path.join(os.path.dirname(__file__), "../skyrl-train/data/fleet_booking_sample.json")
    with open(sample_path) as f:
        sample_tasks = json.load(f)

    print(f"Loaded {len(sample_tasks)} tasks from sample file")

    sample_versions = Counter(t.get('version') for t in sample_tasks)
    print(f"\nVersions in sample file:")
    for v, c in sample_versions.most_common():
        print(f"  {v}: {c}")

    # 3. Find version mismatch
    print("\n" + "=" * 60)
    print("3. Checking version compatibility...")
    print("=" * 60)

    api_versions = set(t.version for t in api_tasks) if api_tasks else set()
    sample_version_set = set(t.get('version') for t in sample_tasks)

    missing = sample_version_set - api_versions
    if missing:
        print(f"\nVersions in sample but NOT in API: {missing}")
    else:
        print("\nAll sample versions exist in API")

    # 4. Try to make environment from API task
    print("\n" + "=" * 60)
    print("4. Testing task.make() from API task...")
    print("=" * 60)

    if api_tasks:
        t = api_tasks[0]
        print(f"Task: {t.key}")
        print(f"  version: {t.version}")
        print(f"  data_id: {t.data_id}")
        print(f"  data_version: {t.data_version}")

        try:
            env = t.make()
            print("SUCCESS - environment created from API task")
            env.close()
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

    # 5. Try to make environment from sample task
    print("\n" + "=" * 60)
    print("5. Testing task.make() from sample file task...")
    print("=" * 60)

    # Load sample via fleet SDK
    sample_fleet_tasks = fleet.load_tasks_from_file(sample_path)
    if sample_fleet_tasks:
        t = sample_fleet_tasks[0]
        print(f"Task: {t.key}")
        print(f"  version: {t.version}")
        print(f"  data_id: {t.data_id}")
        print(f"  data_version: {t.data_version}")

        try:
            env = t.make()
            print("SUCCESS - environment created from sample task")
            env.close()
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
