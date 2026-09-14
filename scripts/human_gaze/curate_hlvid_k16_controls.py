#!/usr/bin/env python3
"""CPU-only, fail-closed curation of two controls and six immutable R2e endpoints."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import tempfile

import numpy as np
import pyarrow.parquet as pq
import yaml

T90_DF5 = 2.015048373
ANSWER = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
PROTOCOL = {"num_video_frames": 128, "num_video_frames_thumbnail": 64,
            "max_tiles_video": 48, "tile_len": 16, "max_new_tokens": 16,
            "torch_dtype": "bfloat16", "max_batch_size_autogaze": 16,
            "max_batch_size_siglip": 32}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_rows(path):
    raw = Path(path).read_bytes()
    require(raw.endswith(b"\n"), f"Missing terminal newline: {path}")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def validate_answers(rows, benchmark):
    require(len(rows) == len(benchmark), "Incomplete answer population")
    expected = {int(row["question_id"]): row for row in benchmark}
    observed = {}
    for row in rows:
        qid = int(row["question_id"])
        require(qid in expected and qid not in observed, "Duplicate/unknown question ID")
        require(all(row[field] == expected[qid][field]
                    for field in ("video_path", "category", "answer")), "Benchmark identity mismatch")
        match = ANSWER.search(row["prediction"].strip())
        letter = match.group(1).upper() if match else ""
        require(row["prediction_letter"] == letter, "Parser mismatch")
        require(isinstance(row["is_correct"], bool) and
                row["is_correct"] == (letter == row["answer"].strip().upper()), "Scoring mismatch")
        observed[qid] = row
    return [observed[qid] for qid in sorted(expected)]


def video_arrays(rows, videos):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_path"]].append(int(row["is_correct"]))
    require(set(grouped) == set(videos), "Video population mismatch")
    return (np.asarray([sum(grouped[v]) for v in videos], dtype=float),
            np.asarray([len(grouped[v]) for v in videos], dtype=float))


def scores(sums, counts):
    return {"question_micro_accuracy": float(np.sum(sums) / np.sum(counts)),
            "macro_video_accuracy": float(np.mean(sums / counts))}


def seed_summary(values):
    require(len(values) == 6, "Seed uncertainty requires all six independent endpoints")
    values = np.asarray(values, dtype=float)
    sd = float(np.std(values, ddof=1))
    mean = float(np.mean(values))
    half = T90_DF5 * sd / np.sqrt(6)
    return {"mean": mean, "seed_sd": sd, "training_seed_ci90_low": float(mean-half),
            "training_seed_ci90_high": float(mean+half), "training_seeds": 6}


def paired_video_interval(differences, counts, *, seed, iterations):
    """Micro resamples cluster sums/counts; macro gives every video equal weight."""
    differences = np.asarray(differences, dtype=float)
    counts = np.asarray(counts, dtype=float)
    require(differences.shape == counts.shape and np.all(counts > 0), "Invalid paired clusters")
    draw = np.random.default_rng(seed).integers(0, len(counts), size=(iterations, len(counts)))
    boot = {"question_micro_accuracy": differences[draw].sum(1) / counts[draw].sum(1),
            "macro_video_accuracy": (differences[draw] / counts[draw]).mean(1)}
    point = {"question_micro_accuracy": float(differences.sum()/counts.sum()),
             "macro_video_accuracy": float((differences/counts).mean())}
    return {metric: {"difference": point[metric], "paired_video_ci90_low": float(np.quantile(values,.05)),
                     "paired_video_ci90_high": float(np.quantile(values,.95))}
            for metric, values in boot.items()}


def scan_events(path, rows, compatibility):
    """Stream the ~400MB action sidecar; retain only small normalized counters."""
    answers = {row["question_id"]: row for row in rows}
    state_counts, started, completed = Counter(), set(), set()
    metadata, seen_questions = [], set()
    first = last = None
    with Path(path).open() as handle:
        for line in handle:
            event = json.loads(line)
            require(event["compatibility_key"] == compatibility, "Event compatibility mismatch")
            state_counts[event["state"]] += 1
            date = event["recorded_at"]
            first = date if first is None else min(first, date)
            last = date if last is None else max(last, date)
            if event["state"] == "attempt_started":
                started.add(event["attempt_id"])
            if event["state"] != "completed_answer":
                continue
            qid = int(event["question_id"])
            require(qid not in seen_questions and event["answer"] == answers[qid], "Duplicate/mismatched durable QA")
            seen_questions.add(qid)
            completed.add(event["attempt_id"])
            decode, context = event["decode"], event["context"]
            require(decode["decode_usable"] and not decode["sampled_read_failure"] and
                    decode["intended_indices"] == decode["effective_indices"] and
                    not decode["reported_index_mismatches"] and decode["tail_padding_count"] == 0,
                    "Material completed-answer decode discrepancy; inspect affected subset")
            require(not context["context_truncated"], "Completed-answer context truncation")
            counters = event["counters"]
            raw = counters["raw_decoder_spatial_actions_per_tile_frame"]
            valid = counters["post_adaptation_valid_patches_per_tile_frame"]
            padded = counters["post_adaptation_padded_slots_per_tile_frame"]
            require(all(x["availability"] == "complete" for x in (raw, valid, padded)), "Missing new-control counters")
            require(raw["min"] == raw["max"] == 16 and valid["mean"] == padded["mean"] == 64,
                    "Exact-K/recovery counter discrepancy")
            row = {"policy": event["policy_label"], "question_id": qid,
                   "qa_key": event["answer"]["qa_key"], "video_key": event["video_key"],
                   "process_warmup_state": event.get("process_warmup_state", "unknown"),
                   "video_frame_cache_state": event.get("video_frame_cache_state", "unknown"),
                   "decoder_actions_mean": raw["mean"], "valid_patches_mean": valid["mean"],
                   "padded_slots_mean": padded["mean"],
                   "expanded_context_length": context["expanded_context_length"],
                   "expanded_visual_tokens": context["expanded_visual_tokens"],
                   "context_truncated": context["context_truncated"]}
            row.update(event["timing"])
            metadata.append(row)
    require(seen_questions == set(answers), "Missing durable completed-answer sidecars")
    require(completed <= started, "Missing attempt-start provenance")
    return metadata, {"state_counts": dict(state_counts), "unclosed_started_attempts": len(started-completed),
                      "first_event_at": first, "last_event_at": last,
                      "attempt_count_is_not_completed_QA_count": True}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_file(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def elapsed_seconds(value):
    day, clock = value.split("-") if "-" in value else (0,value)
    hours, minutes, seconds = map(int, clock.split(":"))
    return int(day)*86400 + hours*3600 + minutes*60 + seconds


def make_figure(path, per_method, contrasts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9, "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(1,3,figsize=(11.2,4.3),layout="constrained")
    colors = {"trained_k16":"#3476a5", "pretrained_exact_k16":"#d28b2c", "stavis_center16":"#458664"}
    names = {"trained_k16":"Human-gaze K16\n6 H100 seeds", "pretrained_exact_k16":"Pretrained K16\nA40", "stavis_center16":"Center16\nA40"}
    for axis, metric, title in zip(axes[:2],("question_micro_accuracy","macro_video_accuracy"),
                                   ("Primary: question micro QA","Sensitivity: macro-video QA")):
        for i, method in enumerate(colors):
            group = [row for row in per_method if row["method"] == method]
            values = [row[metric] for row in group]
            axis.scatter(i+np.linspace(-.09,.09,len(values)),np.asarray(values)*100,s=24,color=colors[method])
            if len(values)==6:
                stat=seed_summary(values)
                axis.errorbar(i,100*stat["mean"], yerr=100*(stat["training_seed_ci90_high"]-stat["mean"]),
                              fmt="D",color=colors[method],capsize=4,markersize=5)
        axis.set_xticks(range(3),[names[m] for m in colors])
        axis.set_ylabel("Accuracy (%)")
        axis.set_title(title)
        axis.grid(axis="y",alpha=.2)
    axis=axes[2]
    for i, row in enumerate(r for r in contrasts if r["metric"]=="question_micro_accuracy"):
        x=100*row["difference"]
        for offset, lo, hi, color, marker, label in (
            (-.09,"training_seed_ci90_low","training_seed_ci90_high","#3476a5","o","90% training-seed t CI"),
            (.09,"paired_video_ci90_low","paired_video_ci90_high","#8a5c9e","s","90% paired-video CI\nconditional on six seeds")):
            axis.errorbar(x,i+offset,xerr=[[x-100*row[lo]],[100*row[hi]-x]],fmt=marker,color=color,
                          capsize=3,label=label if i==0 else None)
    axis.axvline(0,color=".4",lw=1,ls="--")
    axis.set_yticks([0,1],["Human-gaze − pretrained","Human-gaze − Center16"])
    axis.set_xlabel("Micro QA difference (percentage points)")
    axis.set_title("Separate uncertainty sources")
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,loc="outside lower center",ncols=2)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"),dpi=240)
    plt.close(fig)


def curate(config_path, output):
    require(not output.exists(), "Output exists; use a fresh immutable result directory")
    config=yaml.safe_load(config_path.read_text())
    contract=yaml.safe_load(Path(config["execution_contract"]).read_text())
    require(config["analysis"]["confidence_level"] == .90 and contract["analysis"]["confidence_level"] == .90,
            "Frozen confidence-level drift")
    require(config["trained"]["seeds"] == contract["trained_reference"]["valid_seeds"], "Six-seed mapping drift")
    parquet=Path(config["dataset"]["parquet"])
    require(sha256(parquet)==config["dataset"]["sha256"], "Dataset hash mismatch")
    benchmark=sorted(pq.read_table(parquet).to_pylist(),key=lambda r:int(r["question_id"]))
    require([r["question_id"] for r in benchmark]==list(range(268)), "Official population mismatch")
    videos=sorted({r["video_path"] for r in benchmark})
    require(len(videos)==77, "Official video count mismatch")
    prior=Path(config["trained"]["prior_table"])
    require(sha256(prior)==config["trained"]["prior_table_sha256"], "Prior frozen reference table drift")
    with prior.open() as handle:
        expected={int(r["base_seed"]):r for r in csv.DictReader(handle) if int(r["budget"])==16}
    runs, per_method, provenance, telemetry, events = {}, [], [], [], {}
    for method, seed, directory in (
        [("trained_k16",s,Path(config["trained"]["raw_root"])/f"fixedk16_seed{s}") for s in config["trained"]["seeds"]]
        +[(m,None,Path(config["controls"]["raw_root"])/m) for m in config["controls"]["ids"]]):
        summary_path=directory/"summary.json"
        summary=json.loads(summary_path.read_text())
        path=directory/"results.jsonl"
        rows=validate_answers(read_rows(path),benchmark)
        stream_hash=sha256(path)
        require(stream_hash==summary["results_jsonl_sha256"], "Raw result hash mismatch")
        require(summary["dataset_parquet_sha256"]==config["dataset"]["sha256"] and
                summary["benchmark_split"]==config["dataset"]["split"] and
                not summary["human_gaze_split_accessed"], "Split/data identity mismatch")
        entry={"method":method,"base_seed":seed,"raw_directory":str(directory),
               "results_sha256":stream_hash,"summary_sha256":sha256(summary_path),
               "checkpoint_sha256":summary["autogaze_checkpoint_sha256"],"code_commit":summary["code_commit"],
               "results_last_modified_utc":datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
               "summary_last_modified_utc":datetime.fromtimestamp(summary_path.stat().st_mtime,timezone.utc).isoformat()}
        if seed is not None:
            require(stream_hash==expected[seed]["results_sha256"] and
                    summary["autogaze_checkpoint_sha256"]==expected[seed]["checkpoint_sha256"] and
                    summary["base_seed"]==seed and summary["continuation_seed"]==seed+100000,
                    "Trained checkpoint/seed handoff mismatch")
            require(summary["code_commit"]==config["trained"]["evaluation_commit"], "Trained source mismatch")
            protocol=summary["protocol"]
            adapter=summary["fixed_budget_adapter"]
            require(adapter["exact_decoder_actions_per_autogaze_frame"]==16 and
                    adapter["allow_eos"] is False and adapter["allowed_action_ids"]==[69,264], "Trained acquisition drift")
            entry["retrospective_telemetry"]="missing is missing; current-process-only resume counters not full-arm statistics"
        else:
            manifest_path=directory/"run_manifest.json"
            require(sha256(manifest_path)==summary["run_manifest_sha256"], "Control manifest hash mismatch")
            manifest=json.loads(manifest_path.read_text())
            identity=manifest["inference_identity"]
            protocol=identity["protocol"]
            require(summary["code_commit"]==config["controls"]["evaluation_commit"] and not summary["code_dirty"], "Control source drift")
            require(identity["exact_budget"]==16 and protocol["allow_eos"] is False and protocol["fixed_fine_only"], "Control acquisition drift")
            require(json.loads((directory/"preflight_pass.json").read_text())["status"]=="pass", "Missing first-question gate")
            require(sha256(directory/"events.jsonl")==summary["event_output_sha256"], "Event sidecar hash mismatch")
            metadata, events[method]=scan_events(directory/"events.jsonl",rows,manifest["compatibility_key"])
            telemetry.extend(metadata)
            entry.update({"manifest_sha256":sha256(manifest_path),"inference_identity":identity,
                          "preflight_sha256":sha256(directory/"preflight_pass.json"),
                          "admission_sha256":sha256(directory/"admission_record.json"),
                          "event_sha256":summary["event_output_sha256"]})
        require(all(protocol.get(k)==v for k,v in PROTOCOL.items()), "Temporal/spatial/precision protocol drift")
        require(protocol["generation"].startswith("greedy") and protocol["answer_parser"]=="first standalone A/B/C/D letter",
                "Generation/parser drift")
        sums, counts=video_arrays(rows,videos)
        run_scores=scores(sums,counts)
        require(abs(run_scores["question_micro_accuracy"]-summary["accuracy"])<1e-12 and
                abs(run_scores["macro_video_accuracy"]-summary["macro_video_accuracy"])<1e-12, "Summary metric mismatch")
        runs[(method,seed)]=(sums,counts)
        per_method.append({"method":method,"base_seed":seed,"num_correct":int(sums.sum()),
                           **run_scores,"hardware":config["trained" if seed is not None else "controls"]["hardware"]})
        provenance.append(entry)
    analysis=config["analysis"]
    seed_values=[r for r in per_method if r["method"]=="trained_k16"]
    trained_stats={metric:seed_summary([r[metric] for r in seed_values])
                   for metric in ("question_micro_accuracy","macro_video_accuracy")}
    paired_rows, contrast_summary = [], []
    counts=next(iter(runs.values()))[1]
    trained_mean=np.mean([runs[("trained_k16",s)][0] for s in config["trained"]["seeds"]],axis=0)
    for ci, control in enumerate(config["controls"]["ids"]):
        control_sums=runs[(control,None)][0]
        for seed in config["trained"]["seeds"]:
            result=paired_video_interval(runs[("trained_k16",seed)][0]-control_sums,counts,
                                        seed=analysis["bootstrap_seed"]+seed+100*ci,iterations=analysis["bootstrap_iterations"])
            for metric,value in result.items():
                paired_rows.append({"control":control,"base_seed":seed,"metric":metric,**value})
        average=paired_video_interval(trained_mean-control_sums,counts,
                                     seed=analysis["bootstrap_seed"]+1000+ci,iterations=analysis["bootstrap_iterations"])
        control_scores=scores(control_sums,counts)
        for metric,value in average.items():
            seed_stat=seed_summary([r[metric]-control_scores[metric] for r in seed_values])
            contrast_summary.append({"control":control,"metric":metric,**value,
                                     **{k:v for k,v in seed_stat.items() if k not in ("mean",)},
                                     "paired_video_interval_scope":"conditional on evaluated six training seeds"})
    per_video=[]
    for (method,seed),(sums,counts) in runs.items():
        per_video.extend({"method":method,"base_seed":seed,"video_path":video,"questions":int(n),
                          "correct":int(correct),"accuracy":float(correct/n)}
                         for video,correct,n in zip(videos,sums,counts,strict=True))
    with Path(config["resource_segments"]).open() as handle:
        resources=list(csv.DictReader(handle))
    for row in resources:
        row["allocation_seconds"]=elapsed_seconds(row["elapsed"])
    resource_totals={policy:sum(r["allocation_seconds"] for r in resources if r["policy"]==policy)
                     for policy in config["controls"]["ids"]}
    metrics={"schema_version":1,"primary_metric":analysis["primary_metric"],"sensitivity_metric":analysis["sensitivity_metric"],
             "confidence_level":.90,"dataset_questions":268,"dataset_videos":77,"trained_seeds":6,
             "reused_trained_completed_QA":1608,"control_completed_QA":536,"new_QA_this_curation":0,
             "trained":trained_stats,"control_metrics":[r for r in per_method if r["base_seed"] is None],
             "trained_minus_controls":contrast_summary,"events":events,"resource_allocation_seconds":resource_totals,
             "resource_boundary":"All A40 allocation segments, including preemption; not pure generation or cross-hardware latency",
             "uncertainty_boundary":"Training-seed t and paired-video bootstrap intervals are separate, not a combined CI",
             "reuse_qualification":"Healthy bounded processor compatibility and clean shared decode audit support reuse; they do not retrospectively certify every historic decoded read. Missing historic counters stay missing.",
             "claim_boundary":config["claim_boundary"]}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=f".{output.name}-",dir=output.parent))
    for name, rows in (("per_seed.csv",per_method),("paired_per_seed.csv",paired_rows),("contrast_summary.csv",contrast_summary),
                       ("per_video.csv",per_video),("control_per_example_telemetry.csv",sorted(telemetry,key=lambda r:(r["policy"],r["question_id"]))),
                       ("resource_segments.csv",resources)):
        write_csv(temporary/name,rows)
    json_file(temporary/"metrics.json",metrics)
    (temporary/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    make_figure(temporary/"control_comparison",per_method,contrast_summary)
    readme = f"""# R2e exact-K16 HLVid controls: completed descriptive comparison

Run bundle `{output.name}`. Official HLVid test: 268 questions / 77 videos.
Pretrained and tile-local Center16 controls complete 536 unique QA. Six trained
20k K16 endpoints are reused unchanged (1,608 QA); zero new inference in curation.
Protocol: 128 sampled frames, 64 thumbnails, spatial max_tiles_video=48,
NVILA-8B-HD-Video, bfloat16, greedy, exact fine-only K16, no selector EOS.
HLVid official benchmark test is distinct from the protected human-gaze test;
this curation does not access the latter. HLVid has prior historical exposure.

Primary micro QA: trained equal-seed mean {100*trained_stats['question_micro_accuracy']['mean']:.4f}%;
pretrained 50.0000%; Center16 51.1194%. These small differences do not establish
superiority. Macro-video is a declared sensitivity, not a replacement primary.
`contrast_summary.csv` reports separate 90% training-seed Student-t intervals
(six independent endpoints, df5) and paired 77-video cluster bootstrap intervals
(10,000 iterations; conditional on those six seeds). Per-seed video intervals
are in `paired_per_seed.csv`. Neither uncertainty source replaces the other;
no combined interval or equivalence conclusion is asserted.

Human-gaze references ran on H100 NVL; controls on A40. Cross-hardware QA is
explicitly approved as descriptive, but timing is hardware-specific. Allocation
cost includes preempted segments: pretrained {resource_totals['pretrained_exact_k16']/3600:.4f} A40 hours;
Center16 {resource_totals['stavis_center16']/3600:.4f} A40 hours. Logical completed QA
is not the number of event attempts. Missing historical counters remain missing;
current-process-only resumed summaries are not complete resource measurements.
`control_per_example_telemetry.csv` separates acquisition, selector, encoder,
generation/language timing, actual context and valid versus padded patch counts.
Process warmup is not a video-cache-hit label; these controls re-decode per QA.
Long expanded contexts are retained without truncation, not capped to nominal
40,960/32,768 metadata. A clean new decode audit supports, but cannot prove,
all historical reads. No reference QA rerun is warranted merely for telemetry.

Reproduce from repository root with the committed curation config and the
Linux raw paths in manifest.json, using the established healthy CPU interpreter:
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-project /home/stud/latn/miniconda3/envs/vila-autogaze-eval/bin/python scripts/human_gaze/curate_hlvid_k16_controls.py --config experiments/human_gaze/configs/r2e_hlvid_k16_causal_curation_v1.yaml --output-dir <fresh-directory>`.
The script verifies identity/score/hash/population, streams large sidecars,
and refuses existing outputs. Original control admission/preflight and shared
audit identities are embedded in the manifest. Heavy action sidecars remain
on Linux; no videos, frames, model weights or per-example answer caches in Git.
"""
    (temporary/"README.md").write_text(readme)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True).strip()
    json_file(temporary/"manifest.json",{"schema_version":1,"run_id":output.name,"created_at_utc":datetime.now(timezone.utc).isoformat(),
              "curation_commit":commit,"curation_tracked_dirty":bool(dirty),"curation_script_sha256":sha256(Path(__file__)),
              "environment":{"hostname":socket.gethostname(),"python":platform.python_version(),
                             "numpy":np.__version__,"pyarrow":pq.__version__ if hasattr(pq,"__version__") else __import__("pyarrow").__version__,
                             "thread_limits":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
                             "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")},
              "config_sha256":sha256(config_path),"resource_source_sha256":sha256(Path(config["resource_segments"])),
              "resource_source_command":"sacct -D -j 1701485 --starttime 2026-09-01; parent elapsed plus batch peak fields; captured 2026-09-14 01:40 CEST",
              "parquet_sha256":sha256(parquet),"human_gaze_protected_test_accessed":False,"inputs":provenance,
              "outputs":{p.name:sha256(p) for p in sorted(temporary.iterdir()) if p.is_file()}})
    os.replace(temporary,output)
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    curate(args.config,args.output_dir)
