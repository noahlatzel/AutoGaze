# Proxy Sampling Experiments

This family evaluates temporal tile/frame selection for HLVid. It covers:

- uniform HLVid frame-budget baselines,
- thresholded proxy selection,
- score-minus-cost proxy selection,
- dry sweeps that measure sampling behavior without NVILA inference.

The reusable metric implementations live in `autogaze/temporal_sampling/`.
Execution currently uses:

- `scripts/runners/run_hlvid_proxy_dry_sweep.py` for local dry proxy grids.
- `scripts/runners/evaluate_hlvid_proxy_sampling.py` for full NVILA proxy or uniform runs.

The dry sweep is the most recent local experiment path. It intentionally avoids
NVILA generation and aggregates proxy behavior across the full sweep in one
process.

## Current Decision Points

1. Which proxy metrics are worth full NVILA evaluation after dry sweep?
2. Which cost/threshold setting gives a useful quality/compute curve?
3. Whether semantic proxies add enough signal over thumbnail metrics to justify
   encoder cost.

## Baselines

Always compare proxy runs against uniform sampling at similar visited-frame or
visited-tile counts, not only against the same input frame budget.

Useful existing baselines:

- `outputs/hlvid/proxy_sampling/clean_s1_uniform_128/summary.json`
- `outputs/hlvid/proxy_sampling/clean_s1_uniform_256/summary.json`
- `outputs/hlvid/nvila_fixed_delta/delta_*/summary.json`

## Recommended Next Runs

Use dry sweeps first for new metrics or cost schedules. Promote only a small
set of Pareto candidates to full NVILA runs.

## Plotting

Dry-sweep diagnostics:

```bash
python scripts/analysis/plot_dry_proxy_sweep.py \
  --dry-summary outputs/hlvid/proxy_dry_sweeps/full_128_log_fast/sweep_summary.jsonl \
  --output-dir outputs/plots/proxy_dry_sweep
```

Full HLVid/NVILA proxy benchmark plots:

```bash
python scripts/analysis/plot_hlvid_proxy_benchmarks.py \
  --benchmark-roots outputs/hlvid/proxy_sampling_v1_mtv1 outputs/hlvid/proxy_sampling_v2_cost_sweep_mtv1 \
  --output-dir outputs/plots/hlvid_proxy_benchmarks
```

This writes frontier plots plus HLVid-only benchmark diagnostics:

- `frontier_accuracy_vs_visual_tokens.png`
- `frontier_accuracy_vs_visited_frames.png`
- `frontier_accuracy_vs_time.png`
- `accuracy_delta_vs_uniform.png`
- `category_accuracy_delta_vs_uniform.png`
- `efficiency_vs_uniform.png`
- `timing_breakdown.png`
- `per_question_uniform_comparison.png`
- `benchmark_plot_rows.json`
