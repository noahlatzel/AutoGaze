# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check exact generation and rescoring identity for the zero-gated R5 path."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset
from autogaze.models.autogaze import AutoGaze, AutoGazeConfig, AutoGazeImageProcessor


def one_index_per_source(records):
    indices = []
    seen = set()
    for index, record in enumerate(records):
        if record["source"] not in seen:
            seen.add(record["source"])
            indices.append(index)
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/human_gaze/grpo/"
            "r2b_fresh20k_stage3_base440826_seed540826/checkpoint_latest_gaze"
        ),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("experiments/human_gaze/configs/m0_pretrained_exact16_fold1.yaml"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = str(args.checkpoint)
    data = OmegaConf.to_container(OmegaConf.load(args.data_config), resolve=True)
    processor = AutoGazeImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    dataset = AVGazeStavisDataset(
        root=data["dataset_root"],
        manifest_path=data["manifest_path"],
        split="val",
        load_rgb=True,
        load_heatmap=False,
        cell_mass_path=data["cell_mass_path"],
        image_processor=processor,
    )
    indices = one_index_per_source(dataset.records)
    video = next(
        iter(DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False))
    )["video"].cuda()

    control_config = AutoGazeConfig.from_pretrained(checkpoint, local_files_only=True)
    treatment_config = deepcopy(control_config)
    treatment_config.gaze_model_config.temporal_position_encoding = (
        "causal_feature_transport_logit_bias"
    )
    treatment_config.gaze_model_config.feature_transport_temperature = 0.1
    pretrained = AutoGaze.from_pretrained(checkpoint, local_files_only=True)
    model = AutoGaze(treatment_config)
    missing_keys, unexpected_keys = model.load_state_dict(pretrained.state_dict(), strict=False)
    expected_new = {"gazing_model.feature_transport_bias.gate"}
    if set(missing_keys) != expected_new or unexpected_keys:
        raise AssertionError(
            f"Unexpected checkpoint transfer: missing={missing_keys}, unexpected={unexpected_keys}"
        )
    del pretrained
    model = model.cuda().eval()
    module = model.gazing_model.feature_transport_bias
    gate = float(module.gate.detach())
    if gate != 0.0:
        raise AssertionError(f"Expected exact zero gate, observed {gate}")

    allowed = list(range(int(data["fine_action_offset"]), int(data["actions_per_frame"])))
    call = {
        "max_gaze_tokens_each_frame": int(data["exact_budget"]),
        "allowed_token_ids": allowed,
        "generate_only": True,
    }
    with torch.inference_mode():
        delattr(model.gazing_model, "feature_transport_bias")
        control_gaze = model({"video": video}, **call)
        control_score = model(
            {"video": video},
            gazing_info=control_gaze,
            allowed_token_ids=allowed,
            generate_only=False,
        )
        model.gazing_model.feature_transport_bias = module
        treatment_gaze = model({"video": video}, **call)
        treatment_score = model(
            {"video": video},
            gazing_info=control_gaze,
            allowed_token_ids=allowed,
            generate_only=False,
        )

    generation_keys = ("gazing_pos", "if_padded_gazing", "num_gazing_each_frame")
    equality = {
        key: bool(torch.equal(control_gaze[key], treatment_gaze[key]))
        for key in generation_keys
    }
    equality["log_action_probs"] = bool(
        torch.equal(control_score["log_action_probs"], treatment_score["log_action_probs"])
    )
    equality["action_log_probs_all"] = bool(
        torch.equal(
            control_score["action_log_probs_all"],
            treatment_score["action_log_probs_all"],
        )
    )
    if not all(equality.values()):
        raise AssertionError(f"Zero-gated R5 path changed behavior: {equality}")

    report = {
        "schema_version": 1,
        "status": "passed",
        "checkpoint": checkpoint,
        "split": "val",
        "clips": len(indices),
        "one_clip_per_source": True,
        "exact_budget": int(data["exact_budget"]),
        "gate": gate,
        "tensor_equality": equality,
        "expected_new_parameter": sorted(expected_new),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
