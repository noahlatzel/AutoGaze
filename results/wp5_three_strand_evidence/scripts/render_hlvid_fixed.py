"""Render the three-common-seed R2e HLVid budget contrast from compact tables.

No bootstrap or model inference is performed. The primary figure uses the exact
published paired video-cluster intervals; the sensitivity figure has no intervals.
Input is the original R2e result bundle (per_seed.csv, matched_budget_differences.csv,
metrics.json, manifest.json). Optional --cross-check-dir verifies a compact addendum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#009E73"
INK = "#243442"
MUTED = "#5C6872"
GRAY = "#CBD4DC"
GRID = "#E8EDF1"
BUDGETS = (16, 24, 32)
EXPECTED_SEEDS = (440826, 440827, 440828)
DEFAULT_INPUT = Path("experiments/human_gaze/results/r2e_hlvid_nvila_fixed_budget_all_seeds")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text_lf(path, text):
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def numeric(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}")
    return value


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Mismatch {label}: {actual} vs {expected}")


def validate(source, cross_check=None):
    raw = read_csv(source / "per_seed.csv")
    per_seed = {(int(row["budget"]), int(row["base_seed"])): row for row in raw}
    if len(raw) != len(per_seed) or len(raw) != 12:
        raise ValueError("Expected 12 unique published endpoints")
    sets = [{seed for budget, seed in per_seed if budget == k} for k in BUDGETS]
    common = tuple(sorted(set.intersection(*sets)))
    if common != EXPECTED_SEEDS:
        raise ValueError(f"Expected exactly the three shared seeds {EXPECTED_SEEDS}; found {common}")
    if sets != [set(range(440826, 440832)), set(EXPECTED_SEEDS), set(EXPECTED_SEEDS)]:
        raise ValueError("Unexpected available seed matrix")
    for key, row in per_seed.items():
        close(numeric(row, "question_micro_accuracy"), int(row["num_correct"])/268, f"correct/268 {key}")
        if int(row["invalid_predictions"]) != 0 or len(row["checkpoint_sha256"]) != 64:
            raise ValueError(f"Unexpected validation/checkpoint metadata {key}")
        for metric in ("question_micro_accuracy", "macro_video_accuracy"):
            if not 0 <= numeric(row, metric) <= 1:
                raise ValueError(f"Invalid accuracy {key}")
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    if metrics["primary_metric"] != "exact-match question-micro accuracy":
        raise ValueError("Source primary metric changed")
    if tuple(metrics["matched_budget_seeds"]) != common:
        raise ValueError("Metrics common-seed list differs from table")
    means = {metric: {budget: float(np.mean([numeric(per_seed[budget, seed], metric) for seed in common]))
                      for budget in BUDGETS} for metric in ("question_micro_accuracy", "macro_video_accuracy")}
    summaries = {int(row["budget"]): row for row in metrics["matched_budget_summaries"]}
    for budget in BUDGETS:
        if tuple(summaries[budget]["base_seeds"]) != common or int(summaries[budget]["num_seeds"]) != 3:
            raise ValueError("Source matched summary does not use the same three seeds")
        close(means["question_micro_accuracy"][budget], summaries[budget]["question_micro_accuracy_mean"], f"matched mean K{budget}")
    differences = read_csv(source / "matched_budget_differences.csv")
    lookup = {(int(row["budget_a"]), int(row["budget_b"])): row for row in differences}
    if set(lookup) != {(16, 24), (16, 32), (24, 32)} or len(differences) != 3:
        raise ValueError("Unexpected paired comparison matrix")
    source_diffs = {(int(row["budget_a"]), int(row["budget_b"])): row for row in metrics["matched_budget_differences"]}
    for pair, row in lookup.items():
        close(numeric(row, "difference"), means["question_micro_accuracy"][pair[0]]-means["question_micro_accuracy"][pair[1]], f"paired difference {pair}")
        for key in ("difference", "ci90_low", "ci90_high"):
            close(numeric(row, key), float(source_diffs[pair][key]), f"metrics/table {pair} {key}")
        if not numeric(row, "ci90_low") <= 0 <= numeric(row, "ci90_high"):
            raise ValueError("Expected published intervals to include zero; review the updated claim")
        if row["cluster_unit"] != "video" or int(row["num_video_clusters"]) != 77 or int(row["iterations"]) != 10000:
            raise ValueError("Unexpected bootstrap metadata")
    cross_hashes = None
    if cross_check:
        table_dir = cross_check / "source_tables" if (cross_check / "source_tables").is_dir() else cross_check
        compact = read_csv(table_dir / "per_seed.csv")
        compact_keys = {(int(row["budget"]), int(row["base_seed"])) for row in compact}
        if compact_keys != set(per_seed) or len(compact) != len(per_seed):
            raise ValueError("Addendum endpoint matrix differs")
        for row in compact:
            key = (int(row["budget"]), int(row["base_seed"]))
            for metric in ("question_micro_accuracy", "macro_video_accuracy"):
                close(numeric(row, metric), numeric(per_seed[key], metric), f"addendum {key} {metric}")
            if row["checkpoint_sha256"] != per_seed[key]["checkpoint_sha256"]:
                raise ValueError(f"Addendum checkpoint differs {key}")
        compact_differences = read_csv(table_dir / "matched_budget_differences.csv")
        if compact_differences != differences:
            raise ValueError("Addendum paired differences differ")
        cross_hashes = {name: sha256(table_dir/name) for name in ("per_seed.csv", "matched_budget_differences.csv")}
    points = [{"budget": budget, "base_seed": seed, "num_correct": int(per_seed[budget, seed]["num_correct"]),
               "question_micro_accuracy": numeric(per_seed[budget, seed], "question_micro_accuracy"),
               "macro_video_accuracy": numeric(per_seed[budget, seed], "macro_video_accuracy"),
               "checkpoint_sha256": per_seed[budget, seed]["checkpoint_sha256"]}
              for budget in BUDGETS for seed in common]
    checks = {"available_endpoints": len(raw), "displayed_endpoints": len(points), "common_seeds": list(common),
              "displayed_budgets": list(BUDGETS), "questions_per_endpoint": 268, "videos_per_endpoint": 77,
              "correct_count_micro_check": True, "published_mean_check": True, "published_comparison_check": True,
              "all_paired_intervals_include_zero": True, "cross_check_addendum_sha256": cross_hashes,
              "bootstrap_recomputed": False, "new_inference_calls": 0, "means": means, "displayed_points": points}
    return per_seed, lookup, metrics, checks


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "r2e-hlvid-common-seed-figures-v1",
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRAY,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 11, "axes.titleweight": "bold",
                         "grid.color": GRID, "grid.linewidth": .8})


def heading(fig, title, subtitle):
    fig.text(.055, .975, title, va="top", fontsize=14, weight="bold")
    fig.text(.055, .917, subtitle, va="top", fontsize=10, color=MUTED)


def seed_panel(ax, per_seed, metric, title, ylabel, ylim):
    for seed, color, marker in zip(EXPECTED_SEEDS, (BLUE, ORANGE, TEAL), ("o", "s", "^")):
        values = [100*numeric(per_seed[budget, seed], metric) for budget in BUDGETS]
        ax.plot(BUDGETS, values, color=color, marker=marker, markersize=5.5, linewidth=1.1,
                markeredgecolor="white", markeredgewidth=.6, alpha=.88, zorder=3)
    means = [100*np.mean([numeric(per_seed[budget, seed], metric) for seed in EXPECTED_SEEDS]) for budget in BUDGETS]
    ax.plot(BUDGETS, means, color=INK, marker="D", markersize=7, linewidth=1.8,
            markeredgecolor="white", markeredgewidth=.8, zorder=5)
    ax.set_xticks(BUDGETS, [f"K{budget}" for budget in BUDGETS])
    ax.set_xlim(14, 34)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", pad=11)
    ax.set_xlabel("Unique fine actions per AutoGaze frame")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)
    return means


def seed_legend(fig, y):
    handles = [Line2D([0], [0], color=color, marker=marker, linewidth=1, label=f"Seed {seed}")
               for seed, color, marker in zip(EXPECTED_SEEDS, (BLUE, ORANGE, TEAL), ("o", "s", "^"))]
    handles.append(Line2D([0], [0], color=INK, marker="D", linewidth=1.8, label="Mean of these 3 seeds"))
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(.066, y),
               fontsize=9, columnspacing=2.1, handletextpad=.6)


def save(fig, output, stem, caption, provenance, details):
    files = []
    for extension in ("pdf", "svg", "png"):
        path = output / f"{stem}.{extension}"
        metadata = {"pdf": {"CreationDate": None, "ModDate": None}, "svg": {"Date": None}}.get(extension)
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.13, metadata=metadata)
        if extension == "svg":
            # Matplotlib's path data contains insignificant trailing whitespace.
            # Normalize before hashing so Git and cross-host copies preserve it.
            write_text_lf(path, "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n")
        files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    plt.close(fig)
    data = {"figure": stem, "caption": caption, **provenance, **details, "outputs": files}
    write_text_lf(output / f"{stem}.json", json.dumps(data, indent=2)+"\n")
    return data


def primary(per_seed, differences, output, provenance):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.8, 5.55), gridspec_kw={"width_ratios": [1.12, 1]})
    fig.subplots_adjust(left=.075, right=.975, top=.80, bottom=.285, wspace=.43)
    heading(fig, "Fixed spatial budgets on HLVid", "Primary metric: question-micro exact-match accuracy · 268 questions, 77 videos · three common base seeds")
    means = seed_panel(left, per_seed, "question_micro_accuracy", "a  Paired seed endpoints", "Question-micro QA accuracy (%)", (45, 53))
    left.set_yticks([45, 47, 49, 51, 53])
    for budget, mean in zip(BUDGETS, means):
        offset = 12 if budget == 16 else -18
        left.annotate(f"{mean:.2f}%", xy=(budget, mean), xytext=(0, offset), textcoords="offset points",
                      ha="center", color=INK, fontsize=9, zorder=7,
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    right.set_title("b  Paired budget differences", loc="left", pad=11)
    right.axvline(0, color=INK, linewidth=.9, linestyle=(0, (4, 3)))
    pairs = [(16, 24), (16, 32), (24, 32)]
    all_bounds = []
    for y, pair in zip((2, 1, 0), pairs):
        row = differences[pair]
        estimate, low, high = [100*numeric(row, key) for key in ("difference", "ci90_low", "ci90_high")]
        all_bounds.extend((low, high))
        right.errorbar(estimate, y, xerr=[[estimate-low], [high-estimate]], fmt="o", color=BLUE,
                       markersize=5.5, capsize=3, linewidth=1.5, zorder=3)
    right.set_xlim(min(all_bounds)-1.4, max(all_bounds)+1.4)
    right.set_xticks([-6, -3, 0, 3, 6])
    right.set_ylim(-.5, 2.7)
    right.set_yticks([2, 1, 0], [f"K{a} − K{b}" for a, b in pairs])
    right.set_xlabel("Accuracy difference (percentage points)")
    right.tick_params(axis="y", length=0)
    right.spines["left"].set_visible(False)
    right.grid(axis="x", zorder=0)
    right.set_axisbelow(True)
    right.text(.01, .975, "90% paired video-cluster bootstrap intervals", transform=right.transAxes,
               fontsize=8.8, color=MUTED, va="top")
    seed_legend(fig, .135)
    fig.text(.075, .085, "Only seeds 440826–440828 enter both panels. The additional three K16 endpoints are excluded from this contrast.", color=MUTED, fontsize=9)
    fig.text(.075, .041, "All paired intervals include zero; these results establish neither a reliable budget ordering nor equivalence.", color=MUTED, fontsize=9)
    caption = ("Fixed spatial-budget HLVid results for the three common base seeds 440826, 440827 and 440828. "
               "Panel a connects checkpoints sharing the same base seed across K16, K24 and K32; black diamonds are arithmetic means "
               "of these three seeds (49.75%, 47.89% and 49.13%). Accuracy is question-micro exact match over all 268 questions. "
               "Panel b reproduces the published paired video-cluster-bootstrap intervals for the mean correctness differences across "
               "these same seeds: 90% intervals, 10,000 draws, 77 video clusters. The intervals describe video resampling conditional on "
               "the observed three-seed set; they are not intervals from resampling training seeds. All intervals include zero, so no reliable "
               "budget ordering or equivalence is established. K is the exact number of unique fine actions per AutoGaze frame, with EOS disabled. "
               "The evaluation uses 128 uniformly sampled video frames, 64 full thumbnails and maximum spatial tile budget 48. "
               "The additional K16 seeds, unmatched native reference, and pending pretrained/Center16 causal controls are not plotted.")
    return save(fig, output, "hlvid_fixed_common_seeds", caption, provenance,
                {"primary_metric": "question_micro_accuracy", "paired_comparisons": [differences[pair] for pair in pairs],
                 "mean_percent": dict(zip(BUDGETS, means)), "seed_uncertainty_intervals_plotted": False,
                 "bootstrap_recomputed": False, "control_results_plotted": False})


def sensitivity(per_seed, output, provenance):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5), sharey=True)
    fig.subplots_adjust(left=.075, right=.975, top=.79, bottom=.30, wspace=.2)
    heading(fig, "Sensitivity to QA aggregation", "The same nine endpoints and three common seeds · secondary analysis · observed point estimates only")
    micro = seed_panel(axes[0], per_seed, "question_micro_accuracy", "a  Question micro — primary", "QA accuracy (%)", (45, 54))
    macro = seed_panel(axes[1], per_seed, "macro_video_accuracy", "b  Video macro — secondary", "", (45, 54))
    for ax in axes:
        ax.set_yticks([45, 47, 49, 51, 53])
    seed_legend(fig, .15)
    fig.text(.075, .09, "Micro weights every question equally. Macro first averages questions within each video, then weights all 77 videos equally.", color=MUTED, fontsize=9)
    fig.text(.075, .046, "The highest observed mean changes with aggregation. No new uncertainty intervals or budget-ordering claim are added.", color=MUTED, fontsize=9)
    caption = ("Aggregation sensitivity using exactly the same three common seeds and nine checkpoints as the primary fixed-budget comparison. "
               "Panel a repeats question-micro accuracy, the prespecified primary metric. Panel b reports the secondary video-macro accuracy, "
               "which averages questions within a video and then weights the 77 videos equally. Colored lines pair base seeds; black diamonds "
               "are three-seed arithmetic means. The question-micro means are 49.75%, 47.89% and 49.13% for K16, K24 and K32; the corresponding "
               "video-macro means are 48.80%, 48.28% and 51.87%. The highest observed mean therefore changes from K16 to K32 under the secondary "
               "aggregation. These are point estimates without uncertainty intervals and do not establish a reliable ordering. "
               "The additional K16-only seeds and all causal controls are excluded.")
    return save(fig, output, "hlvid_fixed_aggregation_sensitivity", caption, provenance,
                {"role": "secondary sensitivity", "means_percent": {"question_micro": micro, "video_macro": macro},
                 "uncertainty_intervals_plotted": False, "control_results_plotted": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/wp5_three_strand_evidence/figures/hlvid_fixed"))
    parser.add_argument("--cross-check-dir", type=Path, help="Optional addendum directory or its source_tables directory")
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve() or args.input_dir.resolve() in args.output_dir.resolve().parents:
        parser.error("Output must be outside the immutable input bundle; use a separate staging directory before publication.")
    per_seed, differences, metrics, checks = validate(args.input_dir, args.cross_check_dir)
    source_files = ["per_seed.csv", "matched_budget_differences.csv", "metrics.json", "manifest.json", "config.yaml", "README.md"]
    provenance = {"source_bundle": args.input_dir.name,
                  "source_files": {name: sha256(args.input_dir/name) for name in source_files},
                  "source_text_lf_sha256": {name: hashlib.sha256((args.input_dir/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in source_files},
                  "source_hash_policy": "source_files hashes the actual read bytes; source_text_lf_sha256 normalizes CRLF to LF for cross-platform text-content comparison",
                  "renderer": Path(__file__).name, "renderer_sha256": sha256(Path(__file__)),
                  "matplotlib_version": matplotlib.__version__, "common_seeds": list(EXPECTED_SEEDS),
                  "num_seeds_per_budget": 3, "budgets": list(BUDGETS),
                  "protocol_id": metrics["protocol_id"], "scientific_results_recomputed": False,
                  "means_recomputed_from_published_endpoints": True,
                  "addendum_cross_check": checks["cross_check_addendum_sha256"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    rendered = [primary(per_seed, differences, args.output_dir, provenance), sensitivity(per_seed, args.output_dir, provenance)]
    write_text_lf(args.output_dir/"data_checks.json", json.dumps(checks, indent=2)+"\n")
    captions = "# Fixed-budget HLVid figure captions\n\n"+"\n\n".join(f"## {row['figure']}\n\n{row['caption']}" for row in rendered)+"\n"
    write_text_lf(args.output_dir/"figure_captions.md", captions)
    print(json.dumps({"output": str(args.output_dir.resolve()), "figures": [row["figure"] for row in rendered],
                      "common_seeds": list(EXPECTED_SEEDS), "mean_accuracies": checks["means"],
                      "checks": "passed; source endpoints, differences, correct counts and optional addendum agree"}, indent=2))


if __name__ == "__main__":
    main()
