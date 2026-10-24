#!/usr/bin/env python3
"""Destroy the Vast.ai instance a gpu_run.py run owns, and nothing else.

Usage (from the repo root):
    uv run python .claude/skills/verify-warpforth/scripts/cleanup.py build/verify-evidence/<run-id>

Reads <run-id>/owned_label.txt and destroys only instances whose label matches it
exactly. Leaves the evidence directory in place. Exit 0 once none remain.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_run import load_api_key  # noqa: E402
from vastai import VastAI  # noqa: E402


def owned(sdk: VastAI, label: str) -> list[int]:
    instances = sdk.show_instances()
    if not isinstance(instances, list):
        msg = "Vast.ai returned an invalid instance listing"
        raise TypeError(msg)
    return [int(i["id"]) for i in instances if isinstance(i, dict) and i.get("label") == label]


def main() -> int:
    if len(sys.argv) != 2:  # noqa: PLR2004
        print(__doc__, file=sys.stderr)
        return 2
    label = (Path(sys.argv[1]) / "owned_label.txt").read_text().strip()
    api_key = load_api_key()
    if not api_key:
        print("VASTAI_API_KEY not set (env or .env)", file=sys.stderr)
        return 2
    sdk = VastAI(api_key, retry=1)
    for _ in range(3):
        ids = owned(sdk, label)
        if not ids:
            print(f"no instances left for label={label}")
            return 0
        for instance_id in ids:
            print(f"destroying instance {instance_id} label={label}")
            sdk.destroy_instance(id=instance_id)
        time.sleep(5)
    print(f"instances still listed for label={label}: {owned(sdk, label)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
