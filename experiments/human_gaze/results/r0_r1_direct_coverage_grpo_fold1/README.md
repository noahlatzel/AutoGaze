# Direct human-coverage GRPO on AV-gaze-STAViS fold 1

Decoder-only GRPO learns a stable, input-conditioned gaze policy. No KL is
selected: at 2,315 updates it reaches 0.3521 validation macro K=16 versus
0.3496 for KL 0.01, with a smaller seed range. A replicated low-rate
continuation to 3,241 cumulative updates reaches 0.3829 and beats Center-16 in
all three seeds.

| Result | Macro K=16 mean | Population SD | Seed range |
| --- | ---: | ---: | ---: |
| No KL, 2,315 updates, validation | 0.3521 | 0.0008 | 0.0018 |
| KL 0.01, 2,315 updates, validation | 0.3496 | 0.0021 | 0.0047 |
| No KL, 3,241 updates, validation | 0.3829 | 0.0014 | 0.0030 |
| Selected checkpoints, protected test | 0.3711 | 0.0014 | 0.0033 |

Protected test Center-16/Prior-16 are 0.3616/0.4320. Every selected seed beats
Center-16, but the mean remains 0.0610 below the leakage-safe train-only source
prior.

The center-collapse diagnostic rejects a static-policy explanation. The
learned dynamic checkpoint obtains 0.3840 validation macro, compared with
0.3689 for Center-16, 0.3642 for the fixed Top-16 cells from the learned
policy's own marginal selection frequency, and 0.2612 when selections come
from a different video of the same source.

A validation-only, single-seed duration probe reaches 0.3996 at 4,167
cumulative updates. This is evidence of diminishing continued optimization,
not a replicated or test-selected result.

See `metrics.json` for exact values, `manifest.json` for provenance and heavy
artifact locations, and `resolved_configs/` for representative resolved run
configurations.
