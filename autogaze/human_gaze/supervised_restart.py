"""Frozen lineage for the six same-seed restarts after the first-save defect."""

import hashlib
import json
from pathlib import Path
import subprocess

from autogaze.human_gaze.supervised_analysis import sha256_file


ORIGINAL_SOURCE = "5a31685d56ec727b9a46db60598a0693fae7e20e"
ORIGINAL_RUN = "20260910-2128_supervised-k16-comparison_5a31685"
ORIGINAL_ROOT = "/storage/user/latn/artifacts/autogaze-supervised-k16/" + ORIGINAL_RUN
REVIEWED_PARENT = "9c089d183cdfd1e0755287e6fa9db67c73a3bb88"
REPAIR_COMMIT = "13179be38351df168d702962d81ebdca85691a33"
FAILED_INVENTORY_PATH = "experiments/human_gaze/results/supervised_k16_restart/failed_attempts.json"
FAILED_INVENTORY_SHA256 = "d19e39245c2fdcaafb713e73747a07dc81708b3dc752006fd4c6b84a21f53f41"
REPAIRED_CODE_SHA256 = {
    "autogaze/train.py": "5966e9252da70f6572b29231dbca72cc74b015988f1796af46d48598dd67432c",
    "autogaze/trainer.py": "ed4daba54ad2e176dbbbe419d4ea0a9772f1507df4b7a7b48c7a2ae27615b9e3",
}
ALLOWED_CHILD_FILES = {
    "autogaze/human_gaze/supervised_restart.py", "autogaze/human_gaze/checkpoint_provenance.py",
    "scripts/human_gaze/verify_supervised_k16_execution.py",
    "scripts/human_gaze/curate_supervised_k16_comparison.py",
    "scripts/human_gaze/export_supervised_k16_seed_bundle.sh",
    "experiments/human_gaze/slurm/run_supervised_k16_comparison_array.sbatch",
    "experiments/human_gaze/slurm/run_supervised_k16_restart_array.sbatch",
    "experiments/human_gaze/configs/supervised_k16_restart_execution.yaml",
    "experiments/human_gaze/configs/supervised_k16_restart_analysis.yaml",
    "experiments/human_gaze/SUPERVISED_K16.md",
    "tests/test_supervised_k16_restart_lineage.py",
}


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def verify_published_restart_source(root: Path, commit: str):
    """Bind admitted source to the tested repair and forbid scientific drift."""
    _git(root, "merge-base", "--is-ancestor", REPAIR_COMMIT, commit)
    changes = _git(root, "diff", "--name-only", f"{REPAIR_COMMIT}..{commit}").decode().splitlines()
    if any(path not in ALLOWED_CHILD_FILES and not path.startswith("experiments/human_gaze/results/supervised_k16_restart/") for path in changes):
        raise ValueError("Restart source changes more than reviewed metadata/execution/analysis")
    for path, expected in REPAIRED_CODE_SHA256.items():
        if hashlib.sha256(_git(root, "show", f"{commit}:{path}")).hexdigest() != expected:
            raise ValueError(f"Restart source does not match the tested repair: {path}")
    return {"original_execution_commit": ORIGINAL_SOURCE, "reviewed_parent_commit": REVIEWED_PARENT,
            "metadata_repair_commit": REPAIR_COMMIT, "repaired_code_sha256": REPAIRED_CODE_SHA256}


def verify_restart_source(root: Path, expected_commit: str):
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    if head != expected_commit or _git(root, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ValueError("Restart execution requires the exact clean admitted source")
    if subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=root, stdout=subprocess.DEVNULL).returncode == 0:
        raise ValueError("Restart execution requires a detached admitted source")
    chain = verify_published_restart_source(root, head)
    smoke_path = root / "experiments/human_gaze/results/supervised_k16_restart/checkpoint_smoke.json"
    if sha256_file(smoke_path) != "fbb39e8f8d81d08a72d182f049b6ee4a2b46ba64da6828338abffba06f1670d5":
        raise ValueError("Restart smoke summary hash drift")
    smoke = json.loads(smoke_path.read_text())
    full_path = Path(smoke["full_receipt"])
    if sha256_file(full_path) != "3c9fbcdd41e8039963e92aa975a80ff37ebfaa09fd03f0ed37767ea9963f08c6":
        raise ValueError("Actual-Trainer smoke receipt hash drift")
    full = json.loads(full_path.read_text())
    if full.get("status") != "pass" or full.get("source_commit") != REPAIR_COMMIT or full.get("optimizer_updates") != 1 or full.get("greedy_exact_k") is not True:
        raise ValueError("Restart lacks successful tested-repair save/reload evidence")
    return {"repository": _git(root, "remote", "get-url", "origin").decode().strip(),
            "execution_commit": head, "implementation_base_commit": "d85c6558bf9e8f02ece1f3516b4709a715ebec63",
            "dirty": False, "detached": True, "source_chain": chain}


def load_failed_attempts(root: Path, *, verify_live: bool = True):
    path = root / FAILED_INVENTORY_PATH
    if sha256_file(path) != FAILED_INVENTORY_SHA256:
        raise ValueError("Frozen failed-attempt inventory hash drift")
    value = json.loads(path.read_text())
    if (value.get("schema_version") != 1 or value.get("status") != "frozen_failed_attempt_inventory"
            or value.get("original_source_commit") != ORIGINAL_SOURCE or value.get("original_run_id") != ORIGINAL_RUN
            or value.get("original_training_root") != ORIGINAL_ROOT or value.get("checkpoint_present") is not False
            or value.get("completed_updates_per_seed") != 100
            or [row.get("base_seed") for row in value.get("seeds", [])] != list(range(440826, 440832))):
        raise ValueError("Frozen failed-attempt inventory identity mismatch")
    for row in value["seeds"]:
        seed, index = row["base_seed"], row["base_seed"] - 440826
        if row.get("continuation_seed") != seed + 100000 or row.get("array_index") != index or row.get("failed_array_element") != f"1702710_{index}":
            raise ValueError("Frozen failed-attempt seed/array mismatch")
        if not verify_live:
            continue
        seed_root = Path(ORIGINAL_ROOT) / f"seed{seed}_train{seed + 100000}"
        for name, template in value["path_templates"].items():
            actual = Path(template.format(seed_root=seed_root, array_index=index))
            if actual.is_symlink() or not actual.is_file() or sha256_file(actual) != row["files_sha256"].get(name):
                raise ValueError(f"Preserved failed-attempt evidence drift: {seed}.{name}")
        precheck = json.loads((seed_root / "admission_precheck.json").read_text())
        if precheck.get("status") != "pass" or precheck.get("base_seed") != seed or precheck.get("source", {}).get("execution_commit") != ORIGINAL_SOURCE:
            raise ValueError("Failed attempt precheck has the wrong actual identity")
        if list(seed_root.glob("stage*/checkpoint*")):
            raise ValueError("Failed-attempt inventory no longer matches checkpoint absence")
    return value


def restart_identity(root: Path, base_seed: int, *, verify_live: bool = True):
    failed = load_failed_attempts(root, verify_live=verify_live)
    rows = [row for row in failed["seeds"] if row["base_seed"] == base_seed]
    if len(rows) != 1:
        raise ValueError("Restart base seed is outside the frozen six-seed matrix")
    row = rows[0]
    return {"kind": "same_seed_fresh_restart_no_recoverable_checkpoint", "original_source_commit": ORIGINAL_SOURCE,
            "original_run_id": ORIGINAL_RUN, "original_training_root": ORIGINAL_ROOT,
            "failed_array_element": row["failed_array_element"], "failed_actual_job_id": row["actual_job_id"],
            "failed_completed_updates": 100, "failed_base_clip_presentations": 400, "failed_nominal_action_rows": 102400,
            "failed_inventory": FAILED_INVENTORY_PATH, "failed_inventory_sha256": FAILED_INVENTORY_SHA256,
            "base_seed": base_seed, "continuation_seed": base_seed + 100000}
