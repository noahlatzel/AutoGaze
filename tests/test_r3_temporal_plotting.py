from pathlib import Path

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES
from scripts.human_gaze.plot_r3_temporal_position import (
    SEEDS,
    plot_controls,
    plot_learning,
    plot_sources_and_gate,
    summarize,
    summarize_history,
)


def fake_runs(delta=0.006):
    runs = {"control": {}, "position": {}}
    for arm in ("control", "position"):
        for seed in SEEDS:
            treatment_delta = delta if arm == "position" else 0.0
            validation = []
            for step in (0, 10000):
                row = {
                    "train_step": step,
                    "coverage_k16_macro_source": 0.45 + treatment_delta * step / 10000,
                    "temporal_position_gate": treatment_delta * step / 10000,
                    "temporal_signal_to_feature_rms": abs(
                        treatment_delta * step / 10000
                    ),
                }
                row.update(
                    {
                        f"coverage_k16_source_{source}": 0.4
                        + treatment_delta * step / 10000
                        for source in STAVIS_SOURCES
                    }
                )
                validation.append(row)
            training = [
                {
                    "train_step": step,
                    "temporal_position_gate": treatment_delta * step / 10000,
                }
                for step in (0, 10000)
            ]
            runs[arm][seed] = {
                "validation": validation,
                "training": training,
                "endpoint": validation[-1],
            }
    return runs


def fake_center():
    methods = ("actual", "center", "selection_frequency_top16", "shuffled")
    return {
        arm: {
            seed: {
                "coverage": {
                    method: {"16": {"macro_source_mean": 0.45 - 0.01 * index}}
                    for index, method in enumerate(methods)
                },
                "diagnostics": {
                    "actual_minus_center_macro": 0.08,
                    "actual_minus_static_topk_macro": 0.02,
                    "actual_minus_shuffled_macro": 0.10,
                    "mean_center16_overlap_fraction": 0.5,
                    "mean_actual_shuffled_overlap_fraction": 0.4,
                    "normalized_selection_entropy": 0.7,
                },
            }
            for seed in SEEDS
        }
        for arm in ("control", "position")
    }


def test_r3_plot_and_pass_summary(tmp_path: Path):
    runs = fake_runs()
    steps, curves = plot_learning(runs, tmp_path / "learning.png")
    source_deltas = plot_sources_and_gate(runs, tmp_path / "sources.png")
    center = fake_center()
    plot_controls(center, tmp_path / "controls.png")
    report = summarize(runs, steps, curves, source_deltas, center)
    assert report["decision"] == "pass"
    assert report["endpoint_mean_paired_delta"] > 0.005
    assert report["endpoint_paired_uncertainty"]["sample_std"] == 0.0
    assert report["norm_controlled"] is True
    assert all(
        (tmp_path / name).is_file()
        for name in ("learning.png", "sources.png", "controls.png")
    )


def test_summarize_history_uses_count_weighted_source_means():
    history = {}
    for arm, offset in (("control", 0.0), ("position", 0.25)):
        history[arm] = {"seeds": {}}
        for seed in SEEDS:
            per_source = {}
            for source_index, source in enumerate(STAVIS_SOURCES):
                per_source[source] = {
                    metric: {"count": source_index + 1, "mean": source_index + offset}
                    for metric in (
                        "normal_vs_reset_jaccard",
                        "normal_vs_alternative_jaccard",
                        "reset_centroid_shift",
                        "alternative_centroid_shift",
                        "reset_minus_normal_coverage",
                        "alternative_minus_normal_coverage",
                        "normal_set_jaccard_step",
                        "normal_centroid_step",
                        "gt_centroid_step",
                        "centroid_velocity_error",
                        "centroid_velocity_error_abrupt",
                        "centroid_velocity_error_other",
                        "selection_change_abrupt",
                        "selection_change_other",
                    )
                }
            history[arm]["seeds"][str(seed)] = {"per_source": per_source}
    report = summarize_history(history)
    reset = report["paired_position_minus_control"]["normal_vs_reset_jaccard"]
    assert reset["mean"] == 0.25
    assert set(reset["by_seed"]) == {str(seed) for seed in SEEDS}
