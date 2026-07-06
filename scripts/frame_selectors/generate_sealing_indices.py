# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate HLVid source-frame indices with Sealing MIF/MDF selectors.

This is an adapter around the sampling logic from declare-lab/Sealing. Upstream
code writes selected frame tensors into HDF5 files for MSVD/MSRVTT-style
datasets; this script emits JSONL records of original HLVid frame indices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import normalize, softmax
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNERS_DIR = REPO_ROOT / "scripts" / "runners"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RUNNERS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNERS_DIR))

from hlvid_common import existing_question_ids, load_rows, uniform_indices  # noqa: E402
from scripts.frame_selectors.common import model_path, selected_index_record  # noqa: E402


DEFAULT_DATASET_ROOT = "/storage/slurm/latn/data/AutoGaze/HLVid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["mif", "mdf"], required=True)
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not load Sealing models; emit uniform indices with the Sealing method label.",
    )
    parser.add_argument(
        "--sort-indices",
        action="store_true",
        help="Sort selected source indices into temporal order before writing JSONL.",
    )

    parser.add_argument(
        "--candidate-stride",
        type=int,
        default=1,
        help="Read every Nth source video frame as the selector candidate sequence.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on selector candidates; candidates are uniform over the full video.",
    )
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--video-backend",
        choices=["av", "cv2"],
        default="av",
        help="Video reader backend. PyAV avoids OpenCV/NumPy ABI issues on this cluster.",
    )

    parser.add_argument(
        "--mdf-model",
        default=model_path("git_base_coco"),
        help="Vision backbone for MDF representative-frame features.",
    )
    parser.add_argument(
        "--mdf-window",
        type=int,
        default=-1,
        help="Sealing MDF suppression/local-similarity window W; -1 uses adaptive W=N/(alpha*K).",
    )
    parser.add_argument("--mdf-alpha", type=float, default=2.5)

    parser.add_argument(
        "--mif-candidates",
        type=int,
        default=512,
        help="Uniform candidate frames captioned and scored by MIF before taking top K.",
    )
    parser.add_argument("--mif-caption-model", default=model_path("git_base_coco"))
    parser.add_argument(
        "--mif-sim-model",
        default=model_path("qa_evaluator"),
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def candidate_indices(frame_count: int, stride: int, max_candidates: int | None) -> List[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if stride <= 0:
        raise ValueError("--candidate-stride must be positive")
    indices = list(range(0, frame_count, stride))
    if not indices or indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    if max_candidates is not None and len(indices) > max_candidates:
        if max_candidates <= 0:
            raise ValueError("--max-candidates must be positive when provided")
        indices = [int(i) for i in uniform_indices(frame_count, max_candidates)]
    return indices


def video_frame_count(video_path: Path, backend: str) -> int:
    if backend == "cv2":
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frame_count > 0:
            return frame_count

    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        count = 0
        for _ in container.decode(stream):
            count += 1
    if count <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    return count


def load_pil_frames(video_path: Path, indices: Sequence[int], backend: str) -> List[Image.Image]:
    if backend == "cv2":
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        frames_by_index = {}
        for index in sorted({int(index) for index in indices}):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_by_index[index] = Image.fromarray(frame)
        cap.release()
    else:
        import av

        wanted = {int(index) for index in indices}
        frames_by_index = {}
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in wanted:
                    frames_by_index[frame_index] = frame.to_image()
                    if len(frames_by_index) == len(wanted):
                        break
    frames = [frames_by_index[int(index)] for index in indices if int(index) in frames_by_index]
    if not frames:
        raise ValueError(f"Could not read requested frames from video: {video_path}")
    while len(frames) < len(indices):
        frames.append(frames[-1])
    return frames


def batches(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def preprocess_frames(processor, frames: Sequence[Image.Image], device: str, dtype: torch.dtype) -> torch.Tensor:
    tensors = processor(images=list(frames), return_tensors="pt")["pixel_values"]
    return tensors.to(device=device, dtype=dtype)


@torch.inference_mode()
def git_vision_features(model, processor, frames: Sequence[Image.Image], args: argparse.Namespace) -> torch.Tensor:
    feature_chunks = []
    dtype = torch_dtype(args.dtype)
    for frame_batch in batches(frames, args.batch_size):
        pixel_values = preprocess_frames(processor, frame_batch, args.device, dtype)
        outputs = model(pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)
        feature_chunks.append(normalize(features.detach().float().cpu()))
    return torch.cat(feature_chunks, dim=0)


def sealing_mdf_candidate_positions(features: torch.Tensor, num_frames: int, window: int, alpha: float) -> List[int]:
    total = int(features.shape[0])
    if total == 0:
        return []
    if window == -1:
        window = int(total // (alpha * num_frames))
    window = max(1, int(window))

    sims = features @ features.transpose(0, 1)
    local_average = torch.zeros(total)
    for index in range(window, total - window):
        subsim = sims[index][index - window : index + window]
        local_average[index] = (subsim.sum() - 1) / max(len(subsim) - 1, 1)

    selected: List[int] = []
    intervals = []

    def push_interval(left: int, right: int) -> None:
        if right <= left:
            return
        values, rel_index = local_average[left:right].topk(1)
        intervals.append((-float(values[0]), left, right, left + int(rel_index[0])))
        intervals.sort()

    first = int(local_average.argmax().item())
    selected.append(first)
    push_interval(0, first - window)
    push_interval(first + window, total)

    while intervals and len(selected) < num_frames:
        _, left_bound, right_bound, index = intervals.pop(0)
        if index in selected:
            continue
        selected.append(index)
        push_interval(left_bound, index - window)
        push_interval(index + window, right_bound)

    if len(selected) < min(num_frames, total):
        for index in torch.argsort(local_average, descending=True).tolist():
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) >= min(num_frames, total):
                break
    while len(selected) < num_frames:
        selected.append(selected[-1])
    return selected[:num_frames]


def select_mdf(video_path: Path, frame_count: int, args: argparse.Namespace, state: dict) -> List[int]:
    candidate_source_indices = candidate_indices(frame_count, args.candidate_stride, args.max_candidates)
    frames = load_pil_frames(video_path, candidate_source_indices, args.video_backend)
    features = git_vision_features(state["model"], state["processor"], frames, args)
    positions = sealing_mdf_candidate_positions(features, args.num_frames, args.mdf_window, args.mdf_alpha)
    return [int(candidate_source_indices[position]) for position in positions]


@torch.inference_mode()
def generate_captions(caption_model, processor, frames: Sequence[Image.Image], args: argparse.Namespace) -> List[str]:
    captions: List[str] = []
    dtype = torch_dtype(args.dtype)
    for frame_batch in batches(frames, args.batch_size):
        pixel_values = preprocess_frames(processor, frame_batch, args.device, dtype)
        input_ids = processor(
            text=["[CLS] "] * len(frame_batch),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(args.device)
        output_ids = caption_model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=30)
        captions.extend(processor.batch_decode(output_ids, skip_special_tokens=True, max_length=50))
    return captions


@torch.inference_mode()
def score_question_captions(tokenizer, sim_model, question: str, captions: Sequence[str], args: argparse.Namespace) -> torch.Tensor:
    score_chunks = []
    for caption_batch in batches(captions, args.batch_size):
        inputs = tokenizer(
            text=[question] * len(caption_batch),
            text_pair=list(caption_batch),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        outputs = sim_model(**inputs)
        score_chunks.append(softmax(outputs[0][:, 0], dim=-1).detach().float().cpu())
    return torch.cat(score_chunks, dim=0)


def select_mif(video_path: Path, row: dict, frame_count: int, args: argparse.Namespace, state: dict) -> List[int]:
    num_candidates = max(args.num_frames, args.mif_candidates)
    candidate_source_indices = candidate_indices(frame_count, 1, num_candidates)
    frames = load_pil_frames(video_path, candidate_source_indices, args.video_backend)
    captions = generate_captions(state["caption_model"], state["caption_processor"], frames, args)
    scores = score_question_captions(state["tokenizer"], state["sim_model"], row["question"], captions, args)
    topk = min(args.num_frames, len(candidate_source_indices))
    positions = scores.topk(topk).indices.tolist()
    selected = [int(candidate_source_indices[int(position)]) for position in positions]
    while len(selected) < args.num_frames:
        selected.append(selected[-1])
    return selected


def init_selector_state(args: argparse.Namespace) -> dict:
    if args.dry_run:
        return {}
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer

    dtype = torch_dtype(args.dtype)
    if args.mode == "mdf":
        processor = AutoProcessor.from_pretrained(args.mdf_model)
        full_model = AutoModelForCausalLM.from_pretrained(args.mdf_model).to(device=args.device, dtype=dtype)
        model = full_model.git.image_encoder
        model.eval()
        return {"processor": processor, "model": model}

    caption_processor = AutoProcessor.from_pretrained(args.mif_caption_model)
    caption_model = AutoModelForCausalLM.from_pretrained(args.mif_caption_model).to(device=args.device, dtype=dtype)
    caption_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.mif_sim_model)
    sim_model = AutoModelForSequenceClassification.from_pretrained(args.mif_sim_model).to(args.device)
    sim_model.eval()
    return {
        "caption_processor": caption_processor,
        "caption_model": caption_model,
        "tokenizer": tokenizer,
        "sim_model": sim_model,
    }


def record_for(row: dict, selected_indices: Sequence[int], frame_count: int, args: argparse.Namespace) -> dict:
    if args.sort_indices:
        selected_indices = sorted(int(index) for index in selected_indices)
    return selected_index_record(
        question_id=str(row["question_id"]),
        video_path=str(row["video_path"]),
        method=f"sealing_{args.mode}",
        num_frames=int(args.num_frames),
        selected_source_indices=selected_indices,
        frame_count=frame_count,
        metadata={
            "mode": args.mode,
            "candidate_stride": args.candidate_stride,
            "max_candidates": args.max_candidates,
            "dry_run": args.dry_run,
            "sort_indices": args.sort_indices,
        },
    )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset_root)[args.start_index :]
    if args.limit is not None:
        rows = rows[: args.limit]
    done_ids = existing_question_ids(output_path) if args.resume else set()

    state = init_selector_state(args)
    with output_path.open("a" if args.resume else "w") as f:
        for row in tqdm(rows, desc=f"sealing_{args.mode}"):
            question_id = str(row["question_id"])
            if question_id in done_ids:
                continue
            video_path = dataset_root / "videos" / row["video_path"]
            frame_count = video_frame_count(video_path, args.video_backend)
            if args.dry_run:
                selected = [int(index) for index in uniform_indices(frame_count, args.num_frames)]
            elif args.mode == "mdf":
                selected = select_mdf(video_path, frame_count, args, state)
            else:
                selected = select_mif(video_path, row, frame_count, args, state)
            f.write(json.dumps(record_for(row, selected, frame_count, args)) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
