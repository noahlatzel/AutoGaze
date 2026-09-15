import subprocess
import sys

import torch

from autogaze.tasks.human_heatmap_coverage import HumanHeatmapCoverage
from scripts.human_gaze.verify_k32_fixed_budget import (
    nested_differences,
    normalized_for_budget_parity,
)


def test_exact_k32_accounting_is_512_unique_fine_actions_per_clip() -> None:
    task = HumanHeatmapCoverage(
        clip_len=16,
        exact_budget=32,
        report_budgets=(1, 2, 4, 8, 16, 24, 32),
    )
    local = torch.arange(32).repeat(16, 1)
    frame_offsets = torch.arange(16).unsqueeze(1) * 265
    positions = (local + 69 + frame_offsets).reshape(1, -1)
    cell_mass = torch.full((1, 16, 196), 1.0 / 196)

    outputs = task({"cell_mass": cell_mass}, {"gazing_pos": positions})

    assert task.gaze_model_kwargs["max_gaze_tokens_each_frame"] == 32
    assert outputs["traj_len_each_reward"] == [512]
    assert outputs["fine_cells"].shape == (1, 16, 32)
    assert all(torch.unique(row).numel() == 32 for row in outputs["fine_cells"][0])
    torch.testing.assert_close(
        outputs["reward"],
        torch.tensor([[32.0 / 196]], dtype=outputs["reward"].dtype),
    )


def test_parity_normalization_allows_only_budget_and_run_identity() -> None:
    k36 = {
        "task": {"exact_budget": 36, "report_budgets": [1, 2, 36]},
        "trainer": {"exp_name": "k36", "gaze_weights": "k36/weights", "resume": "k36"},
        "algorithm": {"kl_coefficient": 0.0},
    }
    k32 = {
        "task": {"exact_budget": 32, "report_budgets": [1, 2, 32]},
        "trainer": {"exp_name": "k32", "gaze_weights": "k32/weights", "resume": "k32"},
        "algorithm": {"kl_coefficient": 0.0},
    }
    assert nested_differences(
        normalized_for_budget_parity(k32), normalized_for_budget_parity(k36)
    ) == []

    k32["algorithm"]["kl_coefficient"] = 0.01
    assert nested_differences(
        normalized_for_budget_parity(k32), normalized_for_budget_parity(k36)
    ) == ["algorithm.kl_coefficient"]


def test_curator_direct_script_entrypoint_imports_repository_modules() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/human_gaze/curate_k32_fixed_budget.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--verification" in completed.stdout
