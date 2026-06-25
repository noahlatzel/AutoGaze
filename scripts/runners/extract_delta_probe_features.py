# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cache frozen AutoGaze hidden states for Experiment 1 delta probes."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import av
import numpy as np
import torch
from tqdm import tqdm

from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor


DEFAULT_AUTOGAZE_MODEL = "/storage/slurm/latn/data/AutoGaze/hf_models/nvidia--AutoGaze"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels", required=True, help="Delta label JSON from generate_delta_labels.py.")
    parser.add_argument("--output", required=True, help="Output .pt feature cache.")
    parser.add_argument(
        "--label-metric",
        default=None,
        help="Proxy metric label to extract. Inferred when the label JSON contains exactly one metadata.proxy_metrics entry.",
    )
    parser.add_argument(
        "--heuristic",
        default=None,
        help="Deprecated alias for --label-metric.",
    )
    parser.add_argument("--model", default=DEFAULT_AUTOGAZE_MODEL, help="AutoGaze checkpoint path or Hugging Face model id.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16", help="Model dtype.")
    parser.add_argument(
        "--representations",
        default="final_gaze",
        help=(
            "Comma-separated representations to cache: final_gaze, mean_gaze, "
            "final_frame_mean_gaze, mean_vision."
        ),
    )
    parser.add_argument("--limit-tiles", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--gazing-ratio", type=float, default=0.75, help="AutoGaze gazing-ratio generation control.")
    parser.add_argument("--task-loss-requirement", type=float, default=0.7, help="AutoGaze task-loss generation control.")
    parser.add_argument("--seed", type=int, default=0, help="Torch/numpy seed.")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def read_tile(path: str, start: int, tile_len: int) -> np.ndarray:
    with av.open(path) as container:
        return read_video_pyav(container, np.arange(start, start + tile_len))


def infer_label_metric(label_data: dict, requested: Optional[str]) -> str:
    if requested:
        return requested
    metrics = label_data.get("metadata", {}).get("proxy_metrics")
    if metrics and len(metrics) == 1:
        return str(metrics[0])
    heuristics = label_data.get("metadata", {}).get("heuristics")
    if heuristics and len(heuristics) == 1:
        return str(heuristics[0])
    raise ValueError("--label-metric is required when the label JSON contains multiple or no metrics.")


def collect_examples(label_data: dict, label_metric: str, limit_tiles: Optional[int]) -> List[dict]:
    deltas = [int(delta) for delta in label_data["metadata"]["deltas"]]
    delta_to_index = {delta: idx for idx, delta in enumerate(deltas)}
    examples = []
    for video in label_data["videos"]:
        for tile in video["tiles"]:
            label = tile["labels"].get(label_metric)
            if label is None:
                continue
            label = int(label)
            if label not in delta_to_index:
                raise ValueError(f"Label delta {label} is not in metadata.deltas={deltas}")
            examples.append(
                {
                    "video_path": video["video_path"],
                    "relative_path": video["relative_path"],
                    "start": tile["start"],
                    "label_delta": label,
                    "label_index": delta_to_index[label],
                    "effective_fps": video.get("effective_fps"),
                    "fps_bucket": video.get("fps_bucket"),
                    "frame_count": video.get("frame_count"),
                    "label_source": video.get("label_source"),
                    "label_source_name": video.get("label_source_name"),
                }
            )
            if limit_tiles is not None and len(examples) >= limit_tiles:
                return examples
    return examples


def last_non_padded_gaze_hidden_index(gazing_info: Dict[str, torch.Tensor], hidden_sequence_length: int) -> int:
    spans = hidden_sequence_spans(gazing_info, hidden_sequence_length)
    gaze_indices = torch.cat([span["non_padded_gaze_indices"] for span in spans if span["non_padded_gaze_indices"].numel() > 0])
    if gaze_indices.numel() == 0:
        raise ValueError("AutoGaze generated no non-padded gaze token for this tile.")
    return int(gaze_indices[-1])


def hidden_sequence_spans(gazing_info: Dict[str, torch.Tensor], hidden_sequence_length: int) -> List[Dict[str, torch.Tensor]]:
    num_gazing_each_frame = gazing_info["num_gazing_each_frame"].detach().cpu().tolist()
    num_frames = len(num_gazing_each_frame)
    total_gaze_tokens = sum(num_gazing_each_frame)
    num_vision_tokens_each_frame = (hidden_sequence_length - total_gaze_tokens) // num_frames
    if num_vision_tokens_each_frame * num_frames + total_gaze_tokens != hidden_sequence_length:
        raise ValueError(
            "Cannot infer per-frame vision-token count from hidden sequence length "
            f"{hidden_sequence_length} and gaze lengths {num_gazing_each_frame}."
        )
    if_padded = gazing_info["if_padded_gazing"][0].detach().cpu().split(num_gazing_each_frame)

    offset = 0
    spans = []
    for frame_idx, num_gazing in enumerate(num_gazing_each_frame):
        vision_indices = torch.arange(offset, offset + num_vision_tokens_each_frame, dtype=torch.long)
        offset += num_vision_tokens_each_frame
        non_padded = torch.nonzero(~if_padded[frame_idx], as_tuple=False).flatten()
        gaze_indices = torch.arange(offset, offset + num_gazing, dtype=torch.long)
        non_padded_gaze_indices = gaze_indices[non_padded] if non_padded.numel() > 0 else torch.empty(0, dtype=torch.long)
        spans.append(
            {
                "vision_indices": vision_indices,
                "gaze_indices": gaze_indices,
                "non_padded_gaze_indices": non_padded_gaze_indices,
            }
        )
        if non_padded.numel() > 0:
            pass
        offset += num_gazing
    return spans


def parse_representations(value: str) -> List[str]:
    representations = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"final_gaze", "mean_gaze", "final_frame_mean_gaze", "mean_vision"}
    unknown = sorted(set(representations) - valid)
    if unknown:
        raise ValueError(f"Unknown representations: {unknown}")
    return representations


def representation_features(
    final_hidden: torch.Tensor,
    gazing_info: Dict[str, torch.Tensor],
    representations: List[str],
) -> Dict[str, torch.Tensor]:
    spans = hidden_sequence_spans(gazing_info, final_hidden.shape[0])
    all_non_padded_gaze = torch.cat(
        [span["non_padded_gaze_indices"] for span in spans if span["non_padded_gaze_indices"].numel() > 0]
    )
    if all_non_padded_gaze.numel() == 0:
        raise ValueError("AutoGaze generated no non-padded gaze token for this tile.")
    all_vision = torch.cat([span["vision_indices"] for span in spans])

    outputs = {}
    if "final_gaze" in representations:
        outputs["final_gaze"] = final_hidden[all_non_padded_gaze[-1]]
    if "mean_gaze" in representations:
        outputs["mean_gaze"] = final_hidden[all_non_padded_gaze].mean(dim=0)
    if "final_frame_mean_gaze" in representations:
        final_frame_gaze = next(
            (
                span["non_padded_gaze_indices"]
                for span in reversed(spans)
                if span["non_padded_gaze_indices"].numel() > 0
            ),
            None,
        )
        if final_frame_gaze is None:
            raise ValueError("AutoGaze generated no non-padded final-frame gaze token.")
        outputs["final_frame_mean_gaze"] = final_hidden[final_frame_gaze].mean(dim=0)
    if "mean_vision" in representations:
        outputs["mean_vision"] = final_hidden[all_vision].mean(dim=0)
    return {key: value.detach().float().cpu() for key, value in outputs.items()}


@torch.inference_mode()
def extract_feature(
    model: AutoGaze,
    processor: AutoGazeImageProcessor,
    example: dict,
    tile_len: int,
    device: str,
    dtype: torch.dtype,
    gazing_ratio: float,
    task_loss_requirement: float,
    representations: List[str],
) -> Dict[str, torch.Tensor]:
    raw_tile = read_tile(example["video_path"], example["start"], tile_len)
    video = transform_video_for_pytorch(raw_tile, processor).unsqueeze(0).to(device=device, dtype=dtype)
    inputs = {"video": video}

    gazing_info = model(
        inputs,
        gazing_ratio=gazing_ratio,
        task_loss_requirement=task_loss_requirement,
        generate_only=True,
    )
    outputs = model.gazing_model(video, gazing_info, output_hidden_states=True)
    final_hidden = outputs.hidden_states[-1][0]
    return representation_features(final_hidden, gazing_info, representations)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    label_path = Path(args.labels)
    label_data = json.loads(label_path.read_text())
    label_metric = infer_label_metric(label_data, args.label_metric or args.heuristic)
    tile_len = int(label_data["metadata"]["tile_len"])
    deltas = [int(delta) for delta in label_data["metadata"]["deltas"]]
    examples = collect_examples(label_data, label_metric, args.limit_tiles)
    representations = parse_representations(args.representations)

    dtype = torch_dtype(args.dtype)
    processor = AutoGazeImageProcessor.from_pretrained(args.model)
    model = AutoGaze.from_pretrained(args.model).to(device=args.device, dtype=dtype)
    model.eval()

    features_by_representation = {representation: [] for representation in representations}
    kept_examples = []
    failures = []
    for idx, example in enumerate(tqdm(examples, desc="tiles")):
        try:
            extracted = extract_feature(
                model=model,
                processor=processor,
                example=example,
                tile_len=tile_len,
                device=args.device,
                dtype=dtype,
                gazing_ratio=args.gazing_ratio,
                task_loss_requirement=args.task_loss_requirement,
                representations=representations,
            )
            for representation in representations:
                features_by_representation[representation].append(extracted[representation])
            kept_examples.append(example)
        except Exception as exc:
            failures.append({"index": idx, "example": example, "error": repr(exc)})

    stacked_features = {
        representation: torch.stack(features) if features else torch.empty(0)
        for representation, features in features_by_representation.items()
    }

    output = {
            "metadata": {
            "labels": str(label_path),
            "label_metric": label_metric,
            "heuristic": label_metric,
            "model": args.model,
            "deltas": deltas,
            "tile_len": tile_len,
            "representations": representations,
            "gazing_ratio": args.gazing_ratio,
            "task_loss_requirement": args.task_loss_requirement,
            "num_failures": len(failures),
            "failures": failures[:100],
        },
        "features": stacked_features[representations[0]],
        "features_by_representation": stacked_features,
        "labels": torch.tensor([example["label_index"] for example in kept_examples], dtype=torch.long),
        "label_deltas": torch.tensor([example["label_delta"] for example in kept_examples], dtype=torch.long),
        "examples": kept_examples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "num_examples": len(kept_examples),
                "num_failures": len(failures),
                "feature_shapes": {
                    representation: list(features.shape)
                    for representation, features in stacked_features.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
