#!/usr/bin/env python3
"""One bounded read-only accounting snapshot, including preempted run segments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RECEIPT_COMMIT = "d5c7ae6f42d8a50a9072e25a2b4b3b11180846dc"
RECEIPT_PATH = "experiments/human_gaze/results/r2e_hlvid_nvila_fixed_budget_all_seeds/scheduler_node_relaxation_receipt_20260909T124648+0200.json"
FAILED_LOG = Path("/storage/user/latn/slurm/logs/gaze-r2f-replay-1700681.out")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    repo = Path(__file__).resolve().parents[2]
    start = datetime.now(timezone.utc)
    command = ["sacct", "-D", "-j", "1690935,1700681", "--starttime=2026-09-01", f"--endtime={datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}", "--parsable2", "--format=JobID%64,JobIDRaw%64,JobName,State,ExitCode,Submit,Start,End,ElapsedRaw,NodeList%64,ReqMem,ReqCPUS,ReqTRES%256,AllocTRES%256,MaxRSS"]
    raw = subprocess.check_output(command, text=True)
    lines = raw.splitlines()
    fields = lines[0].split("|")
    rows = [dict(zip(fields, line.split("|"), strict=True)) for line in lines[1:]]
    parents = [row for row in rows if "." not in row["JobIDRaw"]]
    latest = {}
    for row in parents:
        latest[row["JobID"]] = row
    receipt = subprocess.check_output(["git", "show", f"{RECEIPT_COMMIT}:{RECEIPT_PATH}"], cwd=repo)
    log = FAILED_LOG.read_bytes()
    report = {
        "schema_version": 1,
        "started_at_utc": start.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "accounting_local_timezone": os.environ.get("TZ", "system local; Europe/Berlin on this host"),
        "hostname": platform.node(),
        "command": command,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip()),
        "raw_sacct_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "segments_including_batch_steps": rows,
        "latest_parent_segment_by_array_arm_or_job": latest,
        "array_reserved_gpu_seconds_including_preemption": sum(int(row["ElapsedRaw"]) for row in parents if row["JobIDRaw"] != "1700681"),
        "compute_boundary": "allocated A40 lane walltime, not measured CUDA utilization; batch/extern steps are not double-counted",
        "failed_replay": {
            "job_id": "1700681", "source_commit": "ecd72f151e4ca5066ce48b9ecb0032af1cb06645",
            "log_path": str(FAILED_LOG), "log_sha256": hashlib.sha256(log).hexdigest(),
            "log_mtime_utc": datetime.fromtimestamp(FAILED_LOG.stat().st_mtime, timezone.utc).isoformat(),
            "observations": 0, "nvila_generation_calls": 0,
            "failure": "PosixPath JSON serialization before processor initialization completed",
        },
        "node_relaxation_receipt": {"source_commit": RECEIPT_COMMIT, "source_path": RECEIPT_PATH, "sha256": hashlib.sha256(receipt).hexdigest(), "local_file": "scheduler_node_relaxation_receipt.json"},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "sacct_segments.psv").write_text(raw)
    (args.output_dir / "scheduler_node_relaxation_receipt.json").write_bytes(receipt)
    (args.output_dir / "scheduler_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"latest": latest, "array_reserved_gpu_seconds": report["array_reserved_gpu_seconds_including_preemption"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
