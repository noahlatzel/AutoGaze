# HLVid NVILA Experiments

This family tracks direct NVILA-HD-Video evaluation on HLVid, independent of
the broader proxy-sampling grid. It should contain default AutoGaze runs,
fixed-delta baselines, and any future controls that are primarily about the
NVILA evaluator rather than proxy metric design.

Execution currently uses `scripts/runners/evaluate_hlvid_nvila.py`.

This runner is intentionally separate from
`scripts/runners/evaluate_hlvid_proxy_sampling.py`: it is the smaller control
path for direct NVILA baselines, fixed inter-tile deltas, and dense/fixed frame
sources. Use the proxy-sampling evaluator when the run needs proxy scoring,
score-minus-cost action selection, or detailed proxy timing.

## Existing Baselines

- default AutoGaze: `outputs/hlvid/nvila_default_autogaze/summary.json`
- fixed inter-tile deltas: `outputs/hlvid/nvila_fixed_delta/delta_*/summary.json`

Use these as sanity checks when proxy-sampling changes affect HLVid result
interpretation.
