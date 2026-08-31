#!/usr/bin/env python3
"""Run the established HLVid/NVILA evaluator with exact-K fine-only AutoGaze."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autogaze.human_gaze.hlvid import install_fixed_budget_forward


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exact-budget", type=int, required=True)
    parser.add_argument("--fine-action-offset", type=int, default=69)
    parser.add_argument("--actions-per-frame", type=int, default=265)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument(
        "--legacy-runner-dir",
        default="/home/stud/latn/AutoGaze/scripts/runners",
        help="Directory containing the established evaluate_hlvid_nvila.py runner.",
    )
    return parser.parse_known_args()


def main() -> None:
    wrapper_args, runner_argv = parse_wrapper_args()
    runner_dir = Path(wrapper_args.legacy_runner_dir).resolve()
    if not (runner_dir / "evaluate_hlvid_nvila.py").is_file():
        raise FileNotFoundError(f"Missing established HLVid runner under {runner_dir}")
    sys.path.insert(0, str(runner_dir))

    import evaluate_hlvid_nvila as runner

    original_argv = sys.argv
    sys.argv = [str(runner_dir / "evaluate_hlvid_nvila.py"), *runner_argv]
    try:
        runner_args = runner.parse_args()
    finally:
        sys.argv = original_argv

    checkpoint = Path(runner_args.autogaze_model_id)
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete AutoGaze checkpoint: {checkpoint}")

    real_auto_processor = runner.AutoProcessor
    installed_stats = []

    class FixedBudgetAutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            processor = real_auto_processor.from_pretrained(*args, **kwargs)
            stats = install_fixed_budget_forward(
                processor._autogaze_model,
                exact_budget=wrapper_args.exact_budget,
                fine_action_offset=wrapper_args.fine_action_offset,
                actions_per_frame=wrapper_args.actions_per_frame,
            )
            installed_stats.append(stats)
            return processor

    runner.AutoProcessor = FixedBudgetAutoProcessor
    runner.parse_args = lambda: runner_args
    runner.main()

    if len(installed_stats) != 1:
        raise RuntimeError(f"Expected one patched NVILA processor, found {len(installed_stats)}")
    summary_path = Path(runner_args.summary_output)
    summary = json.loads(summary_path.read_text())
    summary["fixed_budget_adapter"] = {
        "policy_label": wrapper_args.policy_label,
        "exact_decoder_actions_per_autogaze_frame": wrapper_args.exact_budget,
        "fine_action_offset": wrapper_args.fine_action_offset,
        "actions_per_frame": wrapper_args.actions_per_frame,
        "allowed_action_ids": [wrapper_args.fine_action_offset, wrapper_args.actions_per_frame - 1],
        "allow_eos": False,
        "post_resolution_adaptation": installed_stats[0].as_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["fixed_budget_adapter"], indent=2))


if __name__ == "__main__":
    main()
