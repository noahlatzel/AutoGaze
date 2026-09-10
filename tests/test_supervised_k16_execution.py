import argparse
import importlib.util
import json
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/human_gaze/verify_supervised_k16_execution.py"
SPEC = importlib.util.spec_from_file_location("verify_supervised_k16_execution", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_execution_config_is_one_fixed_six_seed_training_arm():
    path = ROOT / "experiments/human_gaze/configs/supervised_k16_execution.yaml"
    config = MODULE.load_execution_config(path)
    assert config["base_seeds"] == list(range(440826, 440832))
    assert config["continuation_seeds"] == list(range(540826, 540832))
    assert config["training"]["stage1"]["updates"] == 2315
    assert config["training"]["stage2"]["updates"] == 17685
    assert config["training"]["base_clip_presentations_per_seed"] == 80000
    assert config["training"]["nominal_action_rows_per_seed"] == 20480000
    assert config["admission"]["sole_owner_task"] == "01a05a0a-3c7e-7053-aa24-5069bff7fa96"
    assert config["admission"]["no_hlvid_in_this_admission"] is True


def test_phase_resume_contracts_bind_sampler_teacher_and_fixed_cursor():
    stage1 = MODULE.expected_resume_contract("stage1", 440826)
    stage2 = MODULE.expected_resume_contract("stage2", 540826)
    assert stage1["trainer_seed"] == stage1["sampler_seed"] == stage1["teacher_seed"] == 440826
    assert stage2["trainer_seed"] == stage2["sampler_seed"] == stage2["teacher_seed"] == 540826
    assert stage1["loader_batches"] == stage2["loader_batches"] == 1852
    assert stage1["gradient_accumulation_steps"] == 4
    assert stage1["lr_schedule"] == "linear_w_warmup"
    assert stage2["lr_schedule"] == "constant"


def test_launcher_serializes_phases_and_never_runs_hlvid():
    path = ROOT / "experiments/human_gaze/slurm/run_supervised_k16_comparison_array.sbatch"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-5%1" in text
    assert "#SBATCH --gres=gpu:a40:1" in text
    assert "#SBATCH --mem=32G" in text
    stage1_train = text.index("--config-name av_gaze_stavis_supervised_k16_stage1")
    stage1_verify = text.index("--stage stage1")
    stage2_train = text.index("--config-name av_gaze_stavis_supervised_k16_stage2")
    stage2_verify = text.index("--stage stage2")
    assert stage1_train < stage1_verify < stage2_train < stage2_verify
    assert "trainer.gaze_weights=\"$stage1_dir/checkpoint_latest_gaze\"" in text
    assert "evaluate_hlvid" not in text
    assert "sbatch " not in text and "scontrol requeue" not in text and "scancel " not in text


def test_completed_seed_manifest_binds_both_phase_receipts_and_resource_files(tmp_path):
    pair = {"base_seed": 440826, "continuation_seed": 540826}
    precheck = {
        "status": "pass", "array_index": 0, **pair,
        "source": {"execution_commit": "abc", "implementation_base_commit": MODULE.IMPLEMENTATION_BASE},
        "sha256": {"manifest": MODULE.MANIFEST_SHA256}, "slurm": {"SLURM_JOB_ID": "1"},
    }
    stage1 = {"status": "pass", "train_step": 2315, **pair}
    stage2 = {"status": "pass", "train_step": 17685, **pair}
    values = {"precheck": precheck, "stage1": stage1, "stage2": stage2}
    paths = {}
    for name, value in values.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value), encoding="utf-8")
    for name in ("stage1_time", "stage2_time", "gpu"):
        paths[name] = tmp_path / f"{name}.txt"
        paths[name].write_text(name, encoding="utf-8")
    output = tmp_path / "execution_manifest.json"
    MODULE.command_manifest(argparse.Namespace(
        run_id="20260910-2117_supervised-k16-comparison_abcdef0",
        precheck=paths["precheck"],
        stage1_verification=paths["stage1"],
        stage2_verification=paths["stage2"],
        stage1_time=paths["stage1_time"],
        stage2_time=paths["stage2_time"],
        gpu_telemetry=paths["gpu"],
        output=output,
    ))
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "training_complete_pending_validation_and_final_sacct"
    assert manifest["fixed_endpoint"]["cumulative_updates"] == 20000
    assert manifest["hlvid_executed"] is False
    assert manifest["final_sacct"] == "pending_after_allocation_exit"
