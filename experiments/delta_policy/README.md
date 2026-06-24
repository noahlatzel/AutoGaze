# Delta-Policy Experiments

This family studies temporal-delta prediction before full on-policy rollout.
It includes pseudo-label generation, frozen AutoGaze feature extraction,
supervised delta probes, and contextual-bandit policies over logged all-action
scores.

Execution currently uses:

- `scripts/runners/generate_delta_labels.py`
- `scripts/maintenance/combine_delta_label_files.py`
- `scripts/runners/extract_delta_probe_features.py`
- `scripts/runners/train_delta_probe.py`
- `scripts/runners/train_delta_bandit.py`

## Research Separation

Each run should state:

- representation: e.g. `mean_vision`, `final_gaze`, runtime metadata,
- action space: usually `delta in {1,2,4,8,16,32}`,
- supervision or reward: threshold pseudo-labels, all-action score reward,
- evaluation: label accuracy, top-2 accuracy, reward, delta error,
- compute tradeoff: feature extraction cost, training cost, eventual rollout cost.

## Current Interpretation

The contextual-bandit smoke shows that the current immediate reward mostly
learns a conservative moderate-skip policy. That is useful but not yet a reason
to wire full rollout into the AutoGaze model.
