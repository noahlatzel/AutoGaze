#!/usr/bin/env python3
"""CPU-only, fail-closed curation of gate F1 (HLVid within-frame selection floor).

Primary: Center16 minus Random16 question-micro accuracy with a paired 77-video
cluster bootstrap (90%). Cross-job references (six trained K16 seeds, R2e
Center16) are descriptive only and are reported as unavailable if missing.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import tempfile

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import yaml

from scripts.human_gaze.curate_hlvid_k16_controls import (
    PROTOCOL,
    json_file,
    paired_video_interval,
    read_rows,
    require,
    scan_events,
    scores,
    sha256,
    validate_answers,
    video_arrays,
    write_csv,
)


def decision(interval):
    if interval["paired_video_ci90_low"] > 0:
        return "sensitive: Center16 above the Random16 floor (90% lower bound > 0)"
    if interval["paired_video_ci90_high"] < 0:
        return "random_better: Random16 above Center16 (90% upper bound < 0)"
    return "inconclusive: within-frame selection sensitivity not shown (no equivalence claim)"


def load_arm(directory, arm, benchmark, config, expected_commit):
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    rows = validate_answers(read_rows(directory / "results.jsonl"), benchmark)
    require(sha256(directory / "results.jsonl") == summary["results_jsonl_sha256"], "Result hash mismatch")
    require(summary["dataset_parquet_sha256"] == config["dataset"]["parquet_sha256"]
            and not summary["human_gaze_split_accessed"], "Dataset identity mismatch")
    require(summary["code_commit"] == expected_commit and summary["code_dirty"] is False,
            f"{arm['id']} source drift")
    require(summary["policy_kind"] == arm["policy_kind"] and summary["policy_label"] == arm["policy_label"],
            "Arm identity mismatch")
    manifest_path = directory / "run_manifest.json"
    require(sha256(manifest_path) == summary["run_manifest_sha256"], "Run manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    identity = manifest["inference_identity"]
    protocol = identity["protocol"]
    require(all(protocol.get(k) == v for k, v in PROTOCOL.items()), "Protocol drift")
    require(identity["exact_budget"] == 16 and protocol["allow_eos"] is False and protocol["fixed_fine_only"],
            "Acquisition drift")
    adapter = summary["fixed_budget_adapter"]
    if arm["policy_kind"] == "random16":
        require(identity["random16_seed"] == arm["random_cells_seed"] == adapter["random16_seed"],
                "Random16 seed drift")
    else:
        require(adapter["center16_fine_cells"] == arm["fine_cells_ordered"], "Center16 cell drift")
    preflight = json.loads((directory / "preflight_pass.json").read_text())
    require(preflight["status"] == "pass" and preflight["policy_kind"] == arm["policy_kind"],
            "Missing first-question gate")
    require(sha256(directory / "events.jsonl") == summary["event_output_sha256"], "Event hash mismatch")
    telemetry, events = scan_events(directory / "events.jsonl", rows, manifest["compatibility_key"])
    provenance = {
        "arm": arm["id"], "raw_directory": str(directory), "results_sha256": summary["results_jsonl_sha256"],
        "summary_sha256": sha256(summary_path), "run_manifest_sha256": summary["run_manifest_sha256"],
        "event_sha256": summary["event_output_sha256"], "compatibility_key": manifest["compatibility_key"],
        "preflight_sha256": sha256(directory / "preflight_pass.json"),
        "admission_sha256": sha256(directory / "admission_record.json"),
        "code_commit": summary["code_commit"], "code_dirty": summary["code_dirty"],
        "environment": summary["environment"],
        "autogaze_checkpoint_sha256": summary["autogaze_checkpoint_sha256"],
        "nvila_snapshot_revision": identity["nvila_snapshot_revision"],
        "nvila_artifact_hashes": identity["nvila_artifact_hashes"],
        "decode_audit": identity["decode_audit"], "inference_code_hashes": identity["inference_code_hashes"],
        "random16_seed": identity.get("random16_seed"), "random16_rule": identity.get("random16_rule"),
        "event_summary": events,
    }
    return rows, telemetry, provenance


def letters(rows):
    return np.asarray([row["prediction_letter"] for row in rows])


def disagreement(a, b):
    return float(np.mean(letters(a) != letters(b)))


def trained_reference(config, benchmark):
    ref = config["trained_reference"]
    table = Path(ref["prior_table"])
    if sha256(table) != ref["prior_table_sha256"]:
        return None, "prior trained table hash drift"
    with table.open() as handle:
        expected = {int(r["base_seed"]): r for r in csv.DictReader(handle) if int(r["budget"]) == 16}
    runs = {}
    for seed in ref["seeds"]:
        path = Path(ref["raw_root"]) / f"fixedk16_seed{seed}" / "results.jsonl"
        if not path.is_file():
            return None, f"missing trained raw results: {path}"
        if sha256(path) != expected[seed]["results_sha256"]:
            return None, f"trained raw hash mismatch for seed {seed}"
        runs[seed] = validate_answers(read_rows(path), benchmark)
    return runs, None


def historical_center16(config, benchmark):
    directory = Path(config["historical_center16"]["directory"])
    try:
        summary = json.loads((directory / "summary.json").read_text())
        require(summary["code_commit"] == config["historical_center16"]["evaluation_commit"],
                "Historical Center16 commit mismatch")
        require(sha256(directory / "results.jsonl") == summary["results_jsonl_sha256"],
                "Historical Center16 hash mismatch")
        return validate_answers(read_rows(directory / "results.jsonl"), benchmark), None
    except (OSError, ValueError, KeyError) as error:
        return None, f"historical Center16 unavailable: {error}"


def curate(config_path, raw_root, run_id, expected_commit, output, slurm_job_id):
    require(not output.exists(), "Output exists; use a fresh immutable result directory")
    config = yaml.safe_load(config_path.read_text())
    analysis = config["analysis"]
    require(analysis["confidence_level"] == 0.90 and analysis["bootstrap_iterations"] == 10000,
            "Frozen analysis drift")
    parquet = Path(config["dataset"]["root"]) / config["dataset"]["parquet"]
    require(sha256(parquet) == config["dataset"]["parquet_sha256"], "Dataset hash mismatch")
    benchmark = sorted(pq.read_table(parquet).to_pylist(), key=lambda r: int(r["question_id"]))
    require([int(r["question_id"]) for r in benchmark] == list(range(268)), "Official population mismatch")
    videos = sorted({r["video_path"] for r in benchmark})
    require(len(videos) == 77, "Official video count mismatch")

    arms = {arm["id"]: arm for arm in config["arms"] if arm.get("status") != "not_run"}
    rows, telemetry, provenance, sums = {}, [], [], {}
    for arm_id, arm in arms.items():
        arm_rows, arm_telemetry, arm_provenance = load_arm(
            raw_root / arm_id, arm, benchmark, config, expected_commit)
        rows[arm_id] = arm_rows
        telemetry.extend(arm_telemetry)
        provenance.append(arm_provenance)
        sums[arm_id], counts = video_arrays(arm_rows, videos)
    per_arm = [{"arm": arm_id, "num_correct": int(sums[arm_id].sum()), **scores(sums[arm_id], counts),
                "invalid_predictions": int(sum(not r["prediction_letter"] for r in rows[arm_id]))}
               for arm_id in arms]

    seed = analysis["bootstrap_seed"]
    primary = paired_video_interval(sums["stavis_center16"] - sums["random16"], counts,
                                    seed=seed, iterations=analysis["bootstrap_iterations"])
    contrasts = [{"contrast": "stavis_center16_minus_random16", "role": "primary" if metric ==
                  analysis["primary_metric"] else "secondary", "metric": metric, **value}
                 for metric, value in primary.items()]

    trained, trained_reason = trained_reference(config, benchmark)
    trained_summary = {"available": trained is not None, "reason": trained_reason}
    if trained is not None:
        trained_sums = {s: video_arrays(r, videos)[0] for s, r in trained.items()}
        mean_sums = np.mean(list(trained_sums.values()), axis=0)
        reference = paired_video_interval(mean_sums - sums["random16"], counts,
                                          seed=seed + 1, iterations=analysis["bootstrap_iterations"])
        contrasts.extend({"contrast": "trained_k16_six_seed_mean_minus_random16", "role": "descriptive_cross_job",
                          "metric": metric, **value} for metric, value in reference.items())
        trained_summary.update({
            "seeds": list(trained), "mean": scores(mean_sums, counts),
            "per_seed": {str(s): scores(v, counts) for s, v in trained_sums.items()},
            "boundary": "cross-job, cross-hardware (H100 vs this allocation), conditional on six seeds; descriptive",
        })

    flips = {"random16_vs_stavis_center16": disagreement(rows["random16"], rows["stavis_center16"])}
    if trained is not None:
        for arm_id in arms:
            values = [disagreement(rows[arm_id], r) for r in trained.values()]
            flips[f"{arm_id}_vs_trained_k16_seed_mean"] = float(np.mean(values))
        pairs = [disagreement(a, b) for i, a in enumerate(trained.values())
                 for j, b in enumerate(trained.values()) if i < j]
        flips["trained_k16_between_seed_mean"] = float(np.mean(pairs))
    historical, historical_reason = historical_center16(config, benchmark)
    reproduction = {"available": historical is not None, "reason": historical_reason}
    if historical is not None:
        hist_sums = video_arrays(historical, videos)[0]
        flips["stavis_center16_rerun_vs_r2e_center16"] = disagreement(rows["stavis_center16"], historical)
        reproduction.update({"r2e_center16": scores(hist_sums, counts),
                             "rerun_minus_r2e_question_micro":
                                 float((sums["stavis_center16"] - hist_sums).sum() / counts.sum()),
                             "boundary": "same code path, different hardware/allocation; nondeterminism check only"})

    primary_interval = primary[analysis["primary_metric"]]
    metrics = {
        "schema_version": 1, "gate": "F1", "run_id": run_id,
        "primary_metric": analysis["primary_metric"], "primary_contrast": analysis["primary_contrast"],
        "confidence_level": 0.90, "bootstrap_iterations": analysis["bootstrap_iterations"],
        "bootstrap_seed": seed, "dataset_questions": 268, "dataset_videos": 77,
        "new_QA": sum(len(r) for r in rows.values()), "arms": per_arm,
        "primary": primary_interval, "decision": decision(primary_interval),
        "contrasts": contrasts, "trained_reference": trained_summary,
        "center16_reproduction": reproduction, "answer_disagreement_rates": flips,
        "thumbnails_only": {"status": "not_run",
                            "reason": next(a["reason"] for a in config["arms"] if a["id"] == "thumbnails_only")},
        "claim_boundary": "Setting-specific HLVid/NVILA mtv48 K16 result; inconclusive is not equivalence.",
    }
    per_question = [{"question_id": int(b["question_id"]), "video_path": b["video_path"], "answer": b["answer"],
                     **{f"{arm_id}_letter": rows[arm_id][i]["prediction_letter"] for arm_id in arms}}
                    for i, b in enumerate(benchmark)]
    per_video = [{"arm": arm_id, "video_path": video, "questions": int(n), "correct": int(c),
                  "accuracy": float(c / n)} for arm_id in arms
                 for video, c, n in zip(videos, sums[arm_id], counts, strict=True)]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    write_csv(temporary / "per_arm.csv", per_arm)
    write_csv(temporary / "contrasts.csv", contrasts)
    write_csv(temporary / "per_video.csv", per_video)
    write_csv(temporary / "per_question_letters.csv", per_question)
    write_csv(temporary / "per_example_telemetry.csv",
              sorted(telemetry, key=lambda r: (r["policy"], r["question_id"])))
    json_file(temporary / "metrics.json", metrics)
    (temporary / "config.yaml").write_text(config_path.read_text())
    acc = {r["arm"]: r["question_micro_accuracy"] for r in per_arm}
    (temporary / "README.md").write_text(f"""# F1 Gaze HLVid selection floor — `{run_id}`

Question: is HLVid accuracy (frozen NVILA-8B-HD-Video, 128 frames, 64 full
thumbnails, max_tiles_video=48, exact fine-only K16, greedy) sensitive to
within-frame selection? Arms ran in one allocation from commit `{expected_commit}`:
Random16 (seed {arms['random16']['random_cells_seed']}, per video/tile/frame uniform
draw at the Center16 boundary) and an unchanged Center16 rerun. Thumbnails-only
was not run (see metrics.json). 268 questions / 77 videos per arm.

Question-micro accuracy: Random16 {100*acc['random16']:.2f}%, Center16 {100*acc['stavis_center16']:.2f}%.
Primary Center16 − Random16: {100*primary_interval['difference']:+.2f} pp, 90% paired
77-video bootstrap [{100*primary_interval['paired_video_ci90_low']:+.2f}, {100*primary_interval['paired_video_ci90_high']:+.2f}] pp
(10,000 draws, seed {seed}). Decision: {metrics['decision']}.

Secondary rows in `contrasts.csv` (macro accuracy; trained six-seed mean −
Random16 is cross-job/cross-hardware and descriptive only). Answer disagreement
rates and the Center16 rerun-vs-R2e reproduction check are in metrics.json.
Per-question letters, per-video accuracy and per-example telemetry are included.
Heavy evidence (events, evidence records, logs) stays at `{raw_root}`.
Reproduce: `scripts/human_gaze/curate_f1_selection_floor.py --config
experiments/human_gaze/configs/f1_hlvid_selection_floor_v1.yaml --raw-root <raw> --run-id {run_id}
--expected-commit {expected_commit} --output-dir <fresh>`.
""")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    gpu_file = raw_root / "gpu.csv"
    json_file(temporary / "manifest.json", {
        "schema_version": 1, "gate": "F1", "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_commit": expected_commit, "curation_commit": commit, "curation_tracked_dirty": dirty,
        "curation_script_sha256": sha256(Path(__file__)), "config_sha256": sha256(config_path),
        "parquet_sha256": sha256(parquet), "heavy_artifact_root": str(raw_root),
        "inference_slurm_job_id": provenance[0]["environment"].get("slurm_job_id"),
        "curation_slurm_job_id": slurm_job_id,
        "gpu": gpu_file.read_text().strip().splitlines() if gpu_file.is_file() else None,
        "human_gaze_protected_test_accessed": False, "inputs": provenance,
        "environment": {"hostname": socket.gethostname(), "python": platform.python_version(),
                        "numpy": np.__version__, "pyarrow": pyarrow.__version__},
        "outputs": {p.name: sha256(p) for p in sorted(temporary.iterdir()) if p.is_file()},
    })
    os.replace(temporary, output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    curate(args.config, args.raw_root, args.run_id, args.expected_commit, args.output_dir,
           os.environ.get("SLURM_JOB_ID"))
