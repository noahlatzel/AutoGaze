# Proxy Sampling Experiments

This family evaluates temporal tile/frame selection for HLVid. It covers:

- uniform HLVid frame-budget baselines,
- thresholded proxy selection,
- score-minus-cost proxy selection,
- dry sweeps that measure sampling behavior without NVILA inference.

The reusable metric implementations live in `autogaze/temporal_sampling/`.
Execution currently uses `scripts/evaluate_hlvid_proxy_sampling.py`,
`scripts/run_hlvid_proxy_dry_sweep.py`, and
`scripts/run_hlvid_proxy_dry_sweep_fast.py`.

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
