import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVALUATOR = load("l2_evaluator", "scripts/human_gaze/evaluate_hlvid_nvila_fixed_budget.py")
CURATION = load("l2_curation", "scripts/human_gaze/curate_l2_consistent_layout.py")
CONFIG = yaml.safe_load((ROOT / "experiments/human_gaze/configs/l2_hlvid_consistent_layout_v1.yaml").read_text())
LAUNCHER = (ROOT / "experiments/human_gaze/slurm/run_l2_consistent_layout.sbatch").read_text()


def test_layout_overrides_are_optional_and_validated():
    assert EVALUATOR.processor_layout_overrides(SimpleNamespace()) == {}
    assert EVALUATOR.processor_layout_overrides(SimpleNamespace(target_scales=None, target_patch_size=None)) == {}
    assert EVALUATOR.processor_layout_overrides(
        SimpleNamespace(target_scales="64,128,224,448", target_patch_size=16)
    ) == {"target_scales": [64, 128, 224, 448], "target_patch_size": 16}
    for bad in (SimpleNamespace(target_scales="64,128,224,448", target_patch_size=None),
                SimpleNamespace(target_scales=None, target_patch_size=16),
                SimpleNamespace(target_scales="448,224,128,64", target_patch_size=16),
                SimpleNamespace(target_scales="64,128,224", target_patch_size=16)):
        with pytest.raises(ValueError):
            EVALUATOR.processor_layout_overrides(bad)


def fake(scales, patch, tower_counts):
    processor = SimpleNamespace(target_scales=scales, target_patch_size=patch)
    tower = SimpleNamespace(num_patch_each_scale=tower_counts)
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(), tower])
    return processor, model


def test_registered_layout_matches_the_tower_and_the_released_default_does_not():
    tower = CONFIG["protocol"]["tower_positions_per_scale"]
    assert tower == [16, 64, 196, 784]
    layout = CONFIG["protocol"]["processor_layout"]
    consistent = EVALUATOR.processor_tower_layout(*fake(layout["target_scales"], layout["target_patch_size"], tower))
    assert consistent["consistent"] and consistent["processor_positions_per_frame"] == 1060
    released = EVALUATOR.processor_tower_layout(*fake([56, 112, 224, 448], 16, tower))
    assert not released["consistent"]
    assert released["processor_positions_per_frame"] == 1038 and released["tower_positions_per_frame"] == 1060
    with pytest.raises(ValueError):
        EVALUATOR.processor_tower_layout(SimpleNamespace(target_scales=[64], target_patch_size=16),
                                         SimpleNamespace(modules=lambda: []))


def test_launcher_passes_the_registered_layout_and_arms():
    layout = CONFIG["protocol"]["processor_layout"]
    scales = ",".join(str(value) for value in layout["target_scales"])
    assert f"--target-scales {scales} --target-patch-size {layout['target_patch_size']}" in LAUNCHER
    assert f"#SBATCH --array=0-{len(CONFIG['arms']) - 1}" in LAUNCHER
    assert f"#SBATCH --gres={CONFIG['resources']['gres']}" in LAUNCHER
    assert f"BUDGET={CONFIG['resources']['cumulative_time_limit_seconds_per_arm']}" in LAUNCHER
    for index, arm in enumerate(CONFIG["arms"]):
        line = next(line for line in LAUNCHER.splitlines() if re.match(rf"\s*{index}\) arm=", line))
        assert f"arm={arm['id']};" in line and f"kind={arm['policy_kind']};" in line
        assert f"label={arm['policy_label']};" in line
        if arm["policy_kind"] == "trained":
            assert f"--base-seed {arm['base_seed']} --continuation-seed {arm['continuation_seed']}" in line
            assert Path(arm["checkpoint"]).parent.name in line
        if arm["policy_kind"] == "random16":
            assert f"--random-cells-seed {arm['random_cells_seed']}" in line


def rows(correct, videos):
    return [{"question_id": index, "video_path": video, "is_correct": bool(value),
             "prediction_letter": "A" if value else "B"}
            for index, (value, video) in enumerate(zip(correct, videos))]


def test_gaze_contrast_averages_the_two_seeds_before_pairing():
    videos = ["v0", "v0", "v1", "v1", "v2", "v2"]
    seed_a = rows([1, 1, 0, 0, 1, 0], videos)
    seed_b = rows([1, 0, 0, 1, 1, 0], videos)
    center = rows([1, 0, 0, 0, 0, 0], videos)
    mean = CURATION.mean_rows([seed_a, seed_b])
    assert [row["is_correct"] for row in mean] == [1.0, 0.5, 0.0, 0.5, 1.0, 0.0]
    result = CURATION.contrast(mean, center, sorted(set(videos)), seed=1, iterations=500)
    assert result["question_micro_accuracy"]["difference"] == pytest.approx((3.0 - 1.0) / 6)
    assert result["macro_video_accuracy"]["difference"] == pytest.approx(np.mean([0.25, 0.25, 0.5]))
    with pytest.raises(ValueError):
        CURATION.mean_rows([seed_a, seed_b[:-1] + [dict(seed_b[-1], question_id=99)]])


def test_decision_rule_follows_the_interval_sign():
    assert CURATION.decide({"paired_video_ci90_low": 0.01, "paired_video_ci90_high": 0.05}, "up", "down") == "up"
    assert CURATION.decide({"paired_video_ci90_low": -0.05, "paired_video_ci90_high": -0.01}, "up", "down") == "down"
    assert CURATION.decide({"paired_video_ci90_low": -0.01, "paired_video_ci90_high": 0.02}, "up", "down").startswith(
        "unresolved")


def test_selection_geometry_reads_the_action_sidecar(tmp_path):
    center = set(CONFIG["arms"][0]["fine_cells_ordered"])
    centre_actions = [69 + cell for cell in sorted(center)]
    other_actions = [69 + cell for cell in range(16)]
    event = {"state": "completed_answer",
             "raw_decoder_calls": [{"decoder_action_ids": [[centre_actions, centre_actions], [other_actions, centre_actions]]}]}
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"state": "attempt_started"}) + "\n" + json.dumps(event) + "\n")
    result = CURATION.selection_geometry(path, center)
    assert result["selected_cells"] == 64
    assert result["share_in_central_block"] == pytest.approx(48 / 64)
    assert result["mean_overlap_with_next_frame"] == pytest.approx((1.0 + 0.0) / 2)


def test_config_registers_two_primaries_and_five_arms():
    assert [arm["id"] for arm in CONFIG["arms"]] == ["center16", "random16", "pretrained", "trained_440826", "trained_440827"]
    assert set(CONFIG["analysis"]["primary_contrasts"]) == {"floor", "gaze"}
    assert CONFIG["analysis"]["confidence_level"] == 0.90 and CONFIG["analysis"]["bootstrap_iterations"] == 10000
    assert set(CONFIG["released_layout_references"]) == {arm["id"] for arm in CONFIG["arms"]}
    assert CONFIG["resources"]["qa_calls_maximum"] == 5 * CONFIG["dataset"]["questions"]
