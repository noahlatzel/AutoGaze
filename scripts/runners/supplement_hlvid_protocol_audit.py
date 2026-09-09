#!/usr/bin/env python3
"""Add runtime and stable-key provenance to one completed WP0 decode audit."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from hlvid_evidence import stable_example_key


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo: Path) -> dict[str, object]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            text=True,
        ).strip()
    )
    return {"repository": str(repo.resolve()), "commit": commit, "dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    output = args.audit_dir / "audit_supplement.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace audit supplement: {output}")
    paths = {
        "decode_audit": args.audit_dir / "decode_audit.jsonl",
        "summary": args.audit_dir / "summary.json",
        "protocol_runtime_manifest": args.audit_dir / "protocol_runtime_manifest.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete protocol audit: {path}")

    summary = json.loads(paths["summary"].read_text())
    manifest = json.loads(paths["protocol_runtime_manifest"].read_text())
    split = str(manifest["split"])
    affected = [
        {
            "parquet_row_index": int(index),
            "example_key": stable_example_key(int(index), split),
        }
        for index in summary.get("affected_question_rows", [])
    ]

    import cv2
    import numpy
    import pyarrow
    from PIL import __version__ as pillow_version

    supplement = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.audit_dir.name,
        "runtime": {
            "hostname": platform.node(),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "opencv_version": cv2.__version__,
            "pillow_version": pillow_version,
            "numpy_version": numpy.__version__,
            "pyarrow_version": pyarrow.__version__,
        },
        "git": git_state(args.repo),
        "artifact_hashes": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "dataset": {
            "split": split,
            "parquet_sha256": manifest["dataset_sha256"],
            "exact_hash_makes_row_indices_immutable": True,
        },
        "affected_examples": affected,
        "reuse_qualification": summary["reuse_qualification"],
        "historical_certification": bool(summary.get("historical_certification")),
        "identity_link": {
            "run_id": args.audit_dir.name,
            "protocol_runtime_manifest_sha256": sha256_file(paths["protocol_runtime_manifest"]),
            "decode_audit_sha256": sha256_file(paths["decode_audit"]),
            "summary_sha256": sha256_file(paths["summary"]),
        },
    }
    output.write_text(json.dumps(supplement, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(supplement, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
