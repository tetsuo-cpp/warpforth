#!/usr/bin/env python3
"""Compile Forth kernels locally and execute them on one rented Vast.ai GPU.

Usage (from the repo root):
    uv run python .claude/skills/verify-warpforth/scripts/gpu_run.py SPEC.json

SPEC.json:
    {
      "arch": "sm_70",                      # optional, default sm_70
      "jobs": [
        {
          "name": "add",                     # evidence subdirectory name
          "source": "path/to/kernel.forth",  # relative to the spec file or absolute
          "params": {"DATA": [1, 2, 3]},     # optional; by header param name
          "grid": [1, 1, 1],                 # optional
          "block": [1, 1, 1],                # optional
          "output_param": 0,                 # optional; index into \\! param list
          "output_count": 4,                 # optional
          "expect": [7, 0, 0, 0]             # optional; prefix of the output to assert
        }
      ]
    }

Writes evidence to build/verify-evidence/<run-id>/ and prints that path.
Exit code 0 only if every job ran and every "expect" matched.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from gpu_test.conftest import Compiler, KernelRunner, VastSession  # noqa: E402


def load_api_key() -> str | None:
    key = os.environ.get("VASTAI_API_KEY")
    if key:
        return key
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            stripped = line.strip().removeprefix("export ").strip()
            if stripped.startswith("VASTAI_API_KEY="):
                return stripped.split("=", 1)[1].strip().strip("'\"")
    return None


def matches(actual: list, expected: list) -> bool:
    if len(actual) < len(expected):
        return False
    for a, e in zip(actual, expected, strict=False):
        if isinstance(e, float) or isinstance(a, float):
            if not math.isclose(a, e, rel_tol=1e-9, abs_tol=1e-9):
                return False
        elif a != e:
            return False
    return True


def main() -> int:
    if len(sys.argv) != 2:  # noqa: PLR2004
        print(__doc__, file=sys.stderr)
        return 2
    spec_path = Path(sys.argv[1]).resolve()
    spec = json.loads(spec_path.read_text())
    arch = spec.get("arch", "sm_70")

    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "build" / "verify-evidence" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=2))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "session.log"), logging.StreamHandler()],
    )

    api_key = load_api_key()
    if not api_key:
        print("VASTAI_API_KEY not set (env or .env)", file=sys.stderr)
        return 2

    compiler = Compiler(arch=arch)
    summary: dict = {"run_id": run_id, "arch": arch, "jobs": []}
    ok = True
    session = VastSession(api_key, arch=arch)
    # Record ownership before launching so cleanup.py can find this run's instance.
    (out / "owned_label.txt").write_text(session.instance_label + "\n")
    print(f"evidence: {out}", flush=True)
    try:
        with session:
            summary["instance_id"] = session.instance_id
            runner = KernelRunner(session, compiler)
            for job in spec["jobs"]:
                job_dir = out / job["name"]
                job_dir.mkdir()
                src_path = Path(job["source"])
                if not src_path.is_absolute():
                    src_path = spec_path.parent / src_path
                source = src_path.read_text()
                (job_dir / "kernel.forth").write_text(source)
                record = {"name": job["name"]}
                try:
                    (job_dir / "kernel.ptx").write_text(compiler.compile_source(source))
                    result = runner.run(
                        forth_source=source,
                        params=job.get("params"),
                        grid=tuple(job.get("grid", (1, 1, 1))),
                        block=tuple(job.get("block", (1, 1, 1))),
                        output_param=job.get("output_param", 0),
                        output_count=job.get("output_count"),
                    )
                    record["output"] = result
                    if "expect" in job:
                        record["expect"] = job["expect"]
                        record["pass"] = matches(result, job["expect"])
                        ok &= record["pass"]
                except Exception:  # noqa: BLE001
                    record["error"] = traceback.format_exc()
                    record["pass"] = False
                    ok = False
                (job_dir / "result.json").write_text(json.dumps(record, indent=2))
                summary["jobs"].append(record)
                print(json.dumps(record), flush=True)
    finally:
        summary["cleanup_verified"] = session.instance_id is None
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"evidence: {out}", flush=True)
    return 0 if ok and summary["cleanup_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
