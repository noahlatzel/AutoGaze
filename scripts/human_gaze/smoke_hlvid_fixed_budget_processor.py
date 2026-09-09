#!/usr/bin/env python3
"""Verify that acquisition telemetry leaves the established processor output unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections.abc import Mapping
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from PIL import Image

from autogaze.human_gaze.hlvid import install_fixed_budget_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--autogaze-model-id", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def install_reference_adapter(model: torch.nn.Module) -> None:
    """Literal pre-instrumentation exact-K16 adapter used by R2e."""
    original_forward = model.forward

    def fixed_forward(_model, inputs, *args, **kwargs):
        kwargs["gazing_ratio"] = None
        kwargs["task_loss_requirement"] = None
        kwargs["max_gaze_tokens_each_frame"] = 16
        kwargs["allowed_token_ids"] = tuple(range(69, 265))
        kwargs["allow_eos"] = False
        return original_forward(inputs, *args, **kwargs)

    model.forward = MethodType(fixed_forward, model)


def recursive_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def update(current: Any, path: str) -> None:
        digest.update(path.encode())
        if isinstance(current, torch.Tensor):
            tensor = current.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(current, Mapping):
            for key in sorted(current):
                update(current[key], f"{path}.{key}")
        elif isinstance(current, (list, tuple)):
            for index, item in enumerate(current):
                update(item, f"{path}[{index}]")
        else:
            digest.update(repr(current).encode())

    update(value, "root")
    return digest.hexdigest()


def tensor_shapes(value: Any) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}

    def visit(current: Any, path: str) -> None:
        if isinstance(current, torch.Tensor):
            result[path] = list(current.shape)
        elif isinstance(current, Mapping):
            for key in sorted(current):
                visit(current[key], f"{path}.{key}")
        elif isinstance(current, (list, tuple)):
            for index, item in enumerate(current):
                visit(item, f"{path}[{index}]")

    visit(value, "root")
    return result


def synthetic_frames() -> list[Image.Image]:
    # Keep this fixture to one spatial tile so the compatibility check fits on
    # the 12-GiB VM GPU. Full 128/64-video plumbing is exercised by the H100
    # control preflight before the benchmark continuation begins.
    height, width = 224, 224
    y, x = np.mgrid[:height, :width]
    image = np.stack(
        ((x % 256), (y % 256), ((x + 2 * y) % 256)), axis=-1
    ).astype(np.uint8)
    return [Image.fromarray(image.copy()) for _ in range(16)]


def load_processor(model_path: Path, checkpoint: Path):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        str(model_path),
        autogaze_model_id=str(checkpoint),
        num_video_frames=16,
        num_video_frames_thumbnail=8,
        max_tiles_video=48,
        gazing_ratio_tile=[0.2] + [0.06] * 15,
        gazing_ratio_thumbnail=1,
        task_loss_requirement_tile=0.6,
        task_loss_requirement_thumbnail=None,
        max_batch_size_autogaze=16,
        trust_remote_code=True,
        use_fast=False,
    )


def main() -> None:
    args = parse_args()
    frames = synthetic_frames()
    prompt_text = "Which option is visible? A. one B. two C. three D. four"

    reference = load_processor(args.model_path, args.autogaze_model_id)
    install_reference_adapter(reference._autogaze_model)
    prompt = f"{reference.tokenizer.video_token}\n\nQuestion: {prompt_text}"
    with torch.inference_mode():
        reference_inputs = reference(text=prompt, videos=[frames], return_tensors="pt")
    reference_digest = recursive_digest(reference_inputs)
    reference_shapes = tensor_shapes(reference_inputs)
    reference_context = int(reference_inputs["input_ids"].shape[1])
    del reference_inputs, reference
    torch.cuda.empty_cache()

    instrumented = load_processor(args.model_path, args.autogaze_model_id)
    trace = install_fixed_budget_forward(instrumented._autogaze_model, exact_budget=16)
    trace.start_example()
    with torch.inference_mode():
        instrumented_inputs = instrumented(text=prompt, videos=[frames], return_tensors="pt")
    acquisition = trace.finish_example()
    instrumented_digest = recursive_digest(instrumented_inputs)
    instrumented_shapes = tensor_shapes(instrumented_inputs)
    instrumented_context = int(instrumented_inputs["input_ids"].shape[1])

    report = {
        "schema_version": 1,
        "status": "pass"
        if reference_digest == instrumented_digest
        and reference_shapes == instrumented_shapes
        and reference_context == instrumented_context
        else "fail",
        "comparison": "literal R2e fixed adapter versus instrumented fixed adapter",
        "compatibility_scope": "small processor-only fixture; full 128/64 plumbing is checked in the H100 run",
        "synthetic_video": {"frames": 16, "thumbnail_frames": 8, "width": 224, "height": 224},
        "reference_processor_output_sha256": reference_digest,
        "instrumented_processor_output_sha256": instrumented_digest,
        "reference_total_context_tokens": reference_context,
        "instrumented_total_context_tokens": instrumented_context,
        "tensor_shapes_equal": reference_shapes == instrumented_shapes,
        "tensor_shapes": instrumented_shapes,
        "instrumented_acquisition": acquisition,
        "model_path": str(args.model_path),
        "autogaze_model_id": str(args.autogaze_model_id),
        "hostname": platform.node(),
        "pid": os.getpid(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
