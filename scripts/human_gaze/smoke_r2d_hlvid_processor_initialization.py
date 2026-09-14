#!/usr/bin/env python3
"""CPU-only real NVILA processor construction; no weights, pixels or generation.

Only the AutoGaze weight loader is replaced by a lightweight constructor fixture.
The actual local AutoProcessor, image processor, tokenizer and remote processor
class exercise the previously failing serialization boundary. This supporting
check does not validate CUDA processing, EOS allocation or historic QA tensors.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import transformers
from transformers import AutoProcessor

from autogaze.models.autogaze import AutoGaze
from scripts.human_gaze.replay_r2d_hlvid_allocation import load_replay_processor, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--autogaze-model-id", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.num_video_frames = 128
    args.num_video_frames_thumbnail = 64
    args.max_tiles_video = 48
    args.max_batch_size_autogaze = 16
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke evidence: {args.output}")

    loads = []
    requested_devices = []

    class ConstructorFixture:
        def to(self, device):
            requested_devices.append(str(device))
            return self

        def eval(self):
            return self

    def fixture_loader(checkpoint, **kwargs):
        loads.append({"checkpoint": str(checkpoint), "type": type(checkpoint).__name__, "kwargs": kwargs})
        return ConstructorFixture()

    with patch.object(AutoGaze, "from_pretrained", side_effect=fixture_loader):
        try:
            AutoProcessor.from_pretrained(
                args.model_path,
                autogaze_model_id=args.autogaze_model_id,
                num_video_frames=128,
                num_video_frames_thumbnail=64,
                max_tiles_video=48,
                trust_remote_code=True,
            )
        except TypeError as error:
            if "PosixPath" not in str(error) or "not JSON serializable" not in str(error):
                raise
            original_failure = str(error)
        else:
            raise AssertionError("The original Path boundary did not reproduce its recorded failure")
        assert not loads, "Original failure must occur before AutoGaze weight loading"
        processor = load_replay_processor(AutoProcessor, args)
        image_config = json.loads(processor.image_processor.to_json_string())
        assert image_config["autogaze_model_id"] == str(args.autogaze_model_id)
        assert processor.autogaze_model_id == str(args.autogaze_model_id)
        assert processor.num_video_frames == 128
        assert processor.num_video_frames_thumbnail == 64
        assert processor.max_tiles_video == 48
        assert loads[0]["type"] == "str" and len(loads) == 1
        assert requested_devices == ["cuda"]  # Fixture records the intent; it allocates nothing.

    repo = Path(__file__).resolve().parents[2]
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "scope": "real_processor_initialization_with_autogaze_weight_loader_fixture",
        "original_path_failure_reproduced": original_failure,
        "real_processor_class": f"{type(processor).__module__}.{type(processor).__name__}",
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip()),
        "model_path": str(args.model_path),
        "checkpoint_path": str(args.autogaze_model_id),
        "processing_nvila_sha256": sha256_file(args.model_path / "processing_nvila.py"),
        "checkpoint_config_sha256": sha256_file(args.autogaze_model_id / "config.json"),
        "num_video_frames": 128,
        "num_video_frames_thumbnail": 64,
        "max_tiles_video": 48,
        "autogaze_load_calls": loads,
        "heavy_autogaze_weight_loader_stubbed": True,
        "cuda_allocation_performed": False,
        "nvila_model_loaded": False,
        "nvila_generation_calls": 0,
        "allocation_semantics_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
