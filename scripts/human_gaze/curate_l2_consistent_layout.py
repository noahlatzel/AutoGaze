#!/usr/bin/env python3
"""CPU-only, fail-closed curation of gate L2 (gaze HLVid rerun, consistent processor layout).

Two primaries, each a paired 77-video cluster bootstrap (90%) on question-micro
accuracy: Center16 minus Random16 (floor) and the mean of the two trained seeds
minus Center16 (gaze). Comparisons with the same arms under the released
processor layout come from other jobs and are descriptive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq
import yaml

from scripts.human_gaze.curate_hlvid_k16_controls import (
    PROTOCOL,
    json_file,
    paired_video_interval,
    read_rows,
    require,
    scores,
    sha256,
    validate_answers,
    video_arrays,
    write_csv,
)

FINE_OFFSET = 69
TRAINED = ("trained_440826", "trained_440827")


def decide(interval: dict, positive: str, negative: str) -> str:
    if interval["paired_video_ci90_low"] > 0:
        return positive
    if interval["paired_video_ci90_high"] < 0:
        return negative
    return "unresolved (no equivalence claim)"


def mean_rows(runs: list[list[dict]]) -> list[dict]:
    """Per-question mean correctness over runs, as pseudo-rows for the paired bootstrap."""
    require(all([row["question_id"] for row in run] == [row["question_id"] for row in runs[0]] for run in runs),
            "Seed runs are not aligned")
    return [{"question_id": rows[0]["question_id"], "video_path": rows[0]["video_path"],
             "is_correct": float(np.mean([bool(row["is_correct"]) for row in rows]))}
            for rows in zip(*runs)]


def video_sums(rows: list[dict], videos: list[str]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[float]] = {video: [] for video in videos}
    for row in rows:
        grouped[row["video_path"]].append(float(row["is_correct"]))
    return (np.asarray([sum(grouped[video]) for video in videos], dtype=float),
            np.asarray([len(grouped[video]) for video in videos], dtype=float))


def contrast(a: list[dict], b: list[dict], videos: list[str], *, seed: int, iterations: int) -> dict:
    """a minus b with the shared paired video bootstrap."""
    sums_a, counts = video_sums(a, videos)
    sums_b, counts_b = video_sums(b, videos)
    require(np.array_equal(counts, counts_b), "Arms cover different questions")
    return paired_video_interval(sums_a - sums_b, counts, seed=seed, iterations=iterations)


def letters(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["prediction_letter"] for row in rows])


def selection_geometry(event_path: Path, center_cells: set[int]) -> dict:
    """Stream the action sidecar: share of selected cells in the central block and
    mean overlap of a sub-tile's selection with its next frame."""
    central = total = 0
    overlap_sum = overlap_count = 0
    with event_path.open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("state") != "completed_answer":
                continue
            for call in event["raw_decoder_calls"]:
                for frames in call["decoder_action_ids"]:
                    sets = [frozenset(int(action) - FINE_OFFSET for action in actions) for actions in frames]
                    for cells in sets:
                        central += len(cells & center_cells)
                        total += len(cells)
                    for first, second in zip(sets, sets[1:]):
                        overlap_sum += len(first & second) / len(first)
                        overlap_count += 1
    return {"selected_cells": total, "share_in_central_block": central / total,
            "mean_overlap_with_next_frame": overlap_sum / overlap_count}


def load_arm(directory: Path, arm: dict, benchmark: list[dict], config: dict, expected_commit: str) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    rows = validate_answers(read_rows(directory / "results.jsonl"), benchmark)
    require(sha256(directory / "results.jsonl") == summary["results_jsonl_sha256"], "Result hash mismatch")
    require(summary["dataset_parquet_sha256"] == config["dataset"]["parquet_sha256"]
            and not summary["human_gaze_split_accessed"], "Dataset identity mismatch")
    require(summary["code_commit"] == expected_commit and summary["code_dirty"] is False, f"{arm['id']} source drift")
    require(summary["policy_kind"] == arm["policy_kind"] and summary["policy_label"] == arm["policy_label"],
            "Arm identity mismatch")
    manifest = json.loads((directory / "run_manifest.json").read_text())
    require(sha256(directory / "run_manifest.json") == summary["run_manifest_sha256"], "Run manifest hash mismatch")
    identity = manifest["inference_identity"]
    require(all(identity["protocol"].get(key) == value for key, value in PROTOCOL.items()), "Protocol drift")
    require(identity["exact_budget"] == 16, "Budget drift")
    registered = config["protocol"]["processor_layout"]
    require(identity.get("processor_layout_overrides") == registered, "Processor layout is not the registered one")
    layout = summary["processor_tower_layout"]
    require(layout["consistent"] is True
            and layout["tower_positions_per_scale"] == config["protocol"]["tower_positions_per_scale"],
            "Processor and tower layouts differ")
    if arm["policy_kind"] == "trained":
        require(summary["autogaze_checkpoint_sha256"] == arm["checkpoint_sha256"]
                and summary["base_seed"] == arm["base_seed"], "Trained checkpoint drift")
    if arm["policy_kind"] == "random16":
        require(identity["random16_seed"] == arm["random_cells_seed"], "Random16 seed drift")
    preflight = json.loads((directory / "preflight_pass.json").read_text())
    require(preflight["status"] == "pass" and preflight["policy_kind"] == arm["policy_kind"],
            "Missing first-question gate")
    require(sha256(directory / "events.jsonl") == summary["event_output_sha256"], "Event hash mismatch")
    gpus = sorted({line.split(",")[0].strip() for line in (directory / "gpu.csv").read_text().splitlines() if line.strip()})
    telemetry = summary["per_example_telemetry"]
    return {
        "rows": rows, "summary": summary, "gpus": gpus, "layout": layout,
        "provenance": {
            "arm": arm["id"], "raw_directory": str(directory), "results_sha256": summary["results_jsonl_sha256"],
            "run_manifest_sha256": summary["run_manifest_sha256"], "event_sha256": summary["event_output_sha256"],
            "compatibility_key": manifest["compatibility_key"], "code_commit": summary["code_commit"],
            "autogaze_checkpoint_sha256": summary["autogaze_checkpoint_sha256"], "gpus": gpus,
            "environment": summary["environment"],
            "preflight_sha256": sha256(directory / "preflight_pass.json"),
            "admission_sha256": sha256(directory / "admission_record.json"),
        },
        "telemetry": telemetry,
    }


def reference_rows(path: str, benchmark: list[dict]) -> tuple[list[dict] | None, str | None]:
    file = Path(path)
    if not file.is_file():
        return None, f"missing: {path}"
    try:
        return validate_answers(read_rows(file), benchmark), None
    except Exception as error:  # descriptive reference only; never blocks the primaries
        return None, f"unusable: {type(error).__name__}: {error}"


def curate(config_path: Path, raw_root: Path, run_id: str, expected_commit: str, output: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    require(config["gate"] == "L2", "L2 configuration required")
    dataset = config["dataset"]
    parquet = Path(dataset["root"]) / dataset["parquet"]
    require(sha256(parquet) == dataset["parquet_sha256"], "HLVid parquet changed")
    benchmark = pq.read_table(parquet).to_pylist()
    videos = sorted({row["video_path"] for row in benchmark})
    require(len(benchmark) == dataset["questions"] and len(videos) == dataset["videos"], "HLVid population changed")
    analysis = config["analysis"]
    seed, iterations = int(analysis["bootstrap_seed"]), int(analysis["bootstrap_iterations"])
    arms = {arm["id"]: arm for arm in config["arms"]}
    loaded = {name: load_arm(raw_root / name, arm, benchmark, config, expected_commit) for name, arm in arms.items()}
    rows = {name: value["rows"] for name, value in loaded.items()}
    rows["trained_mean"] = mean_rows([rows[name] for name in TRAINED])

    def acc(name: str) -> dict:
        sums, counts = video_sums(rows[name], videos)
        return scores(sums, counts)

    def diff(a: str, b: str) -> dict:
        return contrast(rows[a], rows[b], videos, seed=seed, iterations=iterations)

    floor = diff("center16", "random16")
    gaze = diff("trained_mean", "center16")
    primaries = {
        "floor": {"contrast": "center16 minus random16", **floor["question_micro_accuracy"],
                  "decision": decide(floor["question_micro_accuracy"],
                                     "compact_block_better: Center16 above Random16 (90% lower bound > 0)",
                                     "random_better: Random16 above Center16 (90% upper bound < 0)")},
        "gaze": {"contrast": "mean of the two trained seeds minus center16", **gaze["question_micro_accuracy"],
                 "decision": decide(gaze["question_micro_accuracy"],
                                    "gaze_trained_better: trained policy above Center16 (90% lower bound > 0)",
                                    "center16_better: Center16 above the trained policy (90% upper bound < 0)")},
    }
    secondary = {
        "floor": floor, "gaze": gaze,
        "pretrained_minus_center16": diff("pretrained", "center16"),
        "trained_minus_pretrained": diff("trained_mean", "pretrained"),
        "trained_minus_random16": diff("trained_mean", "random16"),
        "pretrained_minus_random16": diff("pretrained", "random16"),
    }
    real_arms = list(arms)
    disagreement = {f"{a}__{b}": float(np.mean(letters(rows[a]) != letters(rows[b])))
                    for i, a in enumerate(real_arms) for b in real_arms[i + 1:]}
    center_cells = set(int(cell) for cell in arms["center16"]["fine_cells_ordered"])
    per_arm = []
    for name in real_arms:
        value = loaded[name]
        geometry = selection_geometry(raw_root / name / "events.jsonl", center_cells)
        per_arm.append({"arm": name, **acc(name), "gpus": ";".join(value["gpus"]),
                        "invalid_answers": int(np.sum(letters(rows[name]) == "")), **geometry})
    released = {}
    for name, path in config["released_layout_references"].items():
        reference, problem = reference_rows(path, benchmark)
        if reference is None:
            released[name] = {"available": False, "reason": problem}
            continue
        sums, counts = video_sums(reference, videos)
        released[name] = {
            "available": True, "path": path, "sha256": sha256(Path(path)), **scores(sums, counts),
            "consistent_minus_released": contrast(rows[name], reference, videos, seed=seed, iterations=iterations),
            "answer_letters_changed": float(np.mean(letters(rows[name]) != letters(reference))),
            "note": "other job, possibly other hardware; descriptive",
        }
    gpu_sets = {name: tuple(loaded[name]["gpus"]) for name in real_arms}
    same_gpu = len(set(gpu_sets.values())) == 1 and all(len(value) == 1 for value in gpu_sets.values())
    metrics = {
        "schema_version": 1, "gate": "L2", "run_id": run_id,
        "primaries": primaries, "contrasts_micro_and_macro": secondary,
        "accuracy": {name: acc(name) for name in [*real_arms, "trained_mean"]},
        "answer_letter_disagreement": disagreement,
        "released_layout_references": released,
        "hardware": {"gpus_per_arm": {name: list(value) for name, value in gpu_sets.items()},
                     "all_arms_same_gpu_model": same_gpu},
        "layout": loaded["center16"]["layout"],
        "bootstrap": {"seed": seed, "iterations": iterations, "level": 0.90, "unit": "video"},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(config_path.read_text())
    json_file(output / "metrics.json", metrics)
    write_csv(output / "per_arm.csv", per_arm)
    write_csv(output / "per_question_letters.csv", [
        {"question_id": row["question_id"], "video_path": row["video_path"], "answer": row["answer"],
         **{name: rows[name][index]["prediction_letter"] for name in real_arms}}
        for index, row in enumerate(rows["center16"])])
    write_csv(output / "contrasts.csv", [
        {"contrast": name, "metric": metric, **values}
        for name, block in secondary.items() for metric, values in block.items()])
    check = raw_root / "layout_check.json"
    if check.is_file():
        (output / "layout_check.json").write_text(check.read_text())
    manifest = {
        "schema_version": 1, "gate": "L2", "run_id": run_id,
        "curated_utc": datetime.now(timezone.utc).isoformat(),
        "curation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "execution_commit": expected_commit, "heavy_artifacts": str(raw_root),
        "arms": [loaded[name]["provenance"] for name in real_arms],
        "logical_answerer_calls": {name: len(rows[name]) for name in real_arms},
        "files": {path.name: sha256(path) for path in sorted(output.iterdir())
                  if path.is_file() and path.name not in {"manifest.json", "README.md"}},
    }
    json_file(output / "manifest.json", manifest)

    def pp(block: dict) -> str:
        return (f"{100 * block['difference']:+.2f} pp [{100 * block['paired_video_ci90_low']:+.2f}, "
                f"{100 * block['paired_video_ci90_high']:+.2f}]")

    lines = [
        f"# L2 gaze HLVid rerun with a consistent processor layout: `{run_id}`", "",
        "The processor addresses patches in the vision tower's layout, so the cells a policy selects are the",
        "patches the answerer encodes. Question-micro accuracy on 268 HLVid questions (77 videos), 16 fine cells",
        "per 224 px sub-tile and frame, 128 uniform frames, 64 thumbnails; 90% paired video bootstrap.", "",
        f"**Floor (Center16 minus Random16): {pp(primaries['floor'])} — {primaries['floor']['decision']}.**", "",
        f"**Gaze (two trained seeds minus Center16): {pp(primaries['gaze'])} — {primaries['gaze']['decision']}.**", "",
        "| Arm | Micro (%) | Macro (%) | Cells in central block (%) | GPU |", "|---|---|---|---|---|",
    ]
    for row in per_arm:
        lines.append(f"| {row['arm']} | {100 * row['question_micro_accuracy']:.2f} | "
                     f"{100 * row['macro_video_accuracy']:.2f} | {100 * row['share_in_central_block']:.1f} | {row['gpus']} |")
    lines += ["", "| Arm | Released layout, micro (%) | Consistent minus released (micro) | Letters changed (%) |",
              "|---|---|---|---|"]
    for name, value in released.items():
        if value["available"]:
            lines.append(f"| {name} | {100 * value['question_micro_accuracy']:.2f} | "
                         f"{pp(value['consistent_minus_released']['question_micro_accuracy'])} | "
                         f"{100 * value['answer_letters_changed']:.1f} |")
        else:
            lines.append(f"| {name} | not available ({value['reason']}) | | |")
    lines += ["", "The released-layout rows come from other jobs and partly other hardware and are descriptive.",
              f"All arms on one GPU model: {same_gpu}. Exposed population; the gaze interval is conditional on two seeds.",
              "Protocol and decision rules: `config.yaml`. Heavy outputs: `manifest.json`.", ""]
    (output / "README.md").write_text("\n".join(lines))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = curate(args.config, args.raw_root, args.run_id, args.expected_commit, args.output_dir)
    print(json.dumps(metrics["primaries"], indent=2))


if __name__ == "__main__":
    main()
