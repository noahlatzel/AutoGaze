#!/usr/bin/env python3
"""Stream a present-day HLVid decode audit without loading either model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .hlvid_evidence import canonical_json, legacy_decode_map, stable_video_key
except ImportError:
    from hlvid_evidence import canonical_json, legacy_decode_map, stable_video_key


DEFAULT_ROOT = Path("/storage/slurm/latn/data/AutoGaze/HLVid")
EXPECTED_PARQUET_SHA256 = "ed2a1a47603fd19f5d7fae0db4f9792369f5aff6548f35e0225c3b4bd35e112a"
LEGACY_REVISION = "1eb4d9243f3306cd3a759521d11281ac3c5bb190"
EXPECTED_LOADER_SHA256 = "f7281533b556e508b6c82fc419e66a138d93e151be68932a7a632c8585c6de1e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def requested_indices(decodable_count: int, num_frames: int = 128) -> list[int]:
    if decodable_count < 1 or num_frames < 1:
        raise ValueError("frame count and requested sample count must be positive")
    import numpy as np

    return np.round(np.linspace(0, decodable_count - 1, num_frames)).astype(int).tolist()


class OpenCVReader:
    def __init__(self, path: str | Path):
        import cv2

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise OSError(f"Cannot open video: {path}")
        self.reported_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def can_grab(self, index: int) -> bool:
        self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, index)
        return bool(self.capture.grab())

    def read(self, index: int) -> dict[str, Any]:
        self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = self.capture.read()
        result = {"read_success": bool(success), "rgb_conversion_success": None,
                  "image_conversion_success": None}
        if not success:
            return result
        try:
            frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            result["rgb_conversion_success"] = True
        except Exception as error:
            result["rgb_conversion_success"] = False
            result["fatal_error"] = f"{type(error).__name__}: {error}"
            return result
        try:
            from PIL import Image
            # Execute the same conversion, then release pixels immediately.
            Image.fromarray(frame)
            result["image_conversion_success"] = True
        except Exception as error:
            result["image_conversion_success"] = False
            result["fatal_error"] = f"{type(error).__name__}: {error}"
        return result

    def close(self) -> None:
        self.capture.release()


def audit_video(path: str | Path, *, num_frames: int = 128, reader_factory=OpenCVReader) -> dict[str, Any]:
    """Mirror legacy tail validation, duplicate retry and RGB/PIL conversion.

    The source loader must be checked against this implementation before the
    resulting audit is used to qualify reuse. The manifest records its hash.
    """
    reader = reader_factory(path)
    try:
        reported = reader.reported_count
        validated = reported
        tail_probes = []
        while validated > 0:
            success = reader.can_grab(validated - 1)
            tail_probes.append({"index": validated - 1, "grab_success": success})
            if success:
                break
            validated -= 1
        base = {"reported_frame_count": reported, "validated_frame_count": validated,
                "metadata_count_corrected": reported != validated, "tail_probes": tail_probes}
        if validated <= 0:
            return {**base, "decode_usable": False, "error": "No validated terminal frame",
                    "intended_indices": [], "read_results": [], "effective_indices": []}
        intended = requested_indices(validated, num_frames)
        read_results, successful = [], set()
        for slot, index in enumerate(intended):
            if index in successful:
                continue
            try:
                read = reader.read(index)
            except Exception as error:
                read = {"read_success": None, "fatal_error": f"{type(error).__name__}: {error}"}
            read_results.append({"slot": slot, "index": index, **read})
            if read.get("fatal_error"):
                return {**base, "read_results": read_results, "intended_indices": intended,
                        "decode_usable": False, "effective_indices": [],
                        "sampled_read_failure": any(item["read_success"] is False for item in read_results),
                        "legacy_output_substituted": False, "error": read["fatal_error"]}
            if read["read_success"]:
                successful.add(index)
        mapped = legacy_decode_map(intended, sorted(successful))
        return {**base, "read_results": read_results, **mapped,
                "sampled_read_failure": any(item["read_success"] is False for item in read_results),
                "legacy_output_substituted": bool(mapped["substitutions"])}
    finally:
        reader.close()


def _git_revision(directory: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_video_manifest(parquet: Path, dataset_root: Path, video_column: str) -> tuple[int, list[dict[str, Any]]]:
    """Read only video-path metadata. Column/path selection is explicit."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet)
    if video_column not in parquet_file.schema.names:
        raise ValueError(f"Missing video column {video_column!r}; available: {parquet_file.schema.names}")
    paths = parquet_file.read(columns=[video_column]).column(video_column).to_pylist()
    videos: dict[str, dict[str, Any]] = {}
    for row_index, value in enumerate(paths):
        if not isinstance(value, str) or not value:
            raise ValueError(f"Video column must contain paths; row {row_index} has {type(value).__name__}")
        video_path = Path(value)
        if not video_path.is_absolute():
            video_path = dataset_root / "videos" / video_path
        videos.setdefault(value, {"manifest_path": value, "path": video_path, "question_rows": []})["question_rows"].append(row_index)
    return len(paths), list(videos.values())


def summarize_audit(records: list[dict[str, Any]], question_count: int, loader_verified: bool) -> dict[str, Any]:
    affected = [row for row in records if not row["decode_usable"] or row.get("sampled_read_failure") or row.get("legacy_output_substituted")]
    return {"video_count": len(records), "question_count": question_count,
            "unusable_videos": sum(not row["decode_usable"] for row in records),
            "videos_with_requested_read_failures": sum(bool(row.get("sampled_read_failure")) for row in records),
            "videos_with_metadata_count_correction": sum(bool(row.get("metadata_count_corrected")) for row in records),
            "videos_with_substituted_output": sum(bool(row.get("legacy_output_substituted")) for row in records),
            "affected_question_rows": sorted({i for row in affected for i in row["question_rows"]}),
            "loader_source_verified": loader_verified,
            "reuse_qualification": "inspect_affected_subset" if affected else "no_current_requested_decode_failures",
            "historical_certification": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--video-column", default="video_path")
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--expected-parquet-sha256", default=EXPECTED_PARQUET_SHA256)
    parser.add_argument("--legacy-loader", type=Path, default=Path("/home/stud/latn/AutoGaze/scripts/runners/evaluate_hlvid_nvila.py"))
    parser.add_argument("--expected-loader-sha256", default=EXPECTED_LOADER_SHA256)
    args = parser.parse_args()
    if args.num_frames != 128:
        parser.error("this frozen protocol audit requires --num-frames=128")
    return args


def main() -> int:
    args = parse_args()
    parquet = args.parquet or args.dataset_root / "data" / f"{args.split}-00000-of-00001.parquet"
    parquet_hash = sha256_file(parquet)
    if args.expected_parquet_sha256 and parquet_hash != args.expected_parquet_sha256:
        raise ValueError("Dataset parquet hash differs from the expected frozen manifest")
    loader_hash = sha256_file(args.legacy_loader)
    if args.expected_loader_sha256 and loader_hash != args.expected_loader_sha256:
        raise ValueError("Legacy loader hash differs from the source used to validate this audit")
    loader_verified = loader_hash == EXPECTED_LOADER_SHA256
    question_count, videos = load_video_manifest(parquet, args.dataset_root, args.video_column)
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; completed audit bundles are immutable")
    args.output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "audit_code_revision": _git_revision(Path(__file__).parent),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "legacy_loader_path": str(args.legacy_loader),
        "legacy_loader_sha256": loader_hash,
        "legacy_revision_reported": LEGACY_REVISION,
        "loader_source_verified": loader_verified,
        "dataset_parquet": str(parquet), "dataset_sha256": parquet_hash,
        "split": args.split, "video_column": args.video_column,
        "question_count": question_count, "video_count": len(videos),
        "protocol": {"num_video_frames": args.num_frames, "num_video_frames_thumbnail": 64,
                     "max_tiles_video": 48, "tile_len": 16, "nvila_dtype": "bfloat16",
                     "do_sample": False, "num_beams": 1, "max_new_tokens": 16,
                     "truncation": False},
        "protocol_configuration_source": "frozen_evaluator_contract_not_model_execution",
        "live_model_files_verified": False,
        "nvila_snapshot_revision": "7a5670e20da435d98b0efdc49f9a536f73985152",
        "python": sys.version, "platform": platform.platform(),
        "interpretation": "Present-day decode audit; does not retrospectively certify historical inputs.",
    }
    (args.output_dir / "protocol_runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    records = []
    audit_start = time.perf_counter()
    with (args.output_dir / "decode_audit.jsonl").open("x", encoding="utf-8") as handle:
        for video in videos:
            video_start = time.perf_counter()
            try:
                audit = audit_video(video["path"], num_frames=args.num_frames)
            except Exception as error:
                audit = {"decode_usable": False, "error": f"{type(error).__name__}: {error}"}
            record = {"video_key": stable_video_key(video["manifest_path"], args.split),
                      "manifest_path": video["manifest_path"], "resolved_path": str(video["path"]),
                      "question_rows": video["question_rows"], **audit,
                      "audit_decode_seconds": time.perf_counter() - video_start}
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            records.append(record)
    summary = summarize_audit(records, question_count, loader_verified)
    summary["audit_elapsed_seconds"] = time.perf_counter() - audit_start
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
