# Fixed K16 human-gaze endpoint confirmation

Completed inference-only confirmation of six fixed 20k RL endpoints on the same 663 held-out STAViS clips (67 source/video pairs, six sources). No endpoint was selected using this run. Fine-only exact K16, greedy decoding, 16 frames per clip. Macro coverage aggregates frames → clips → videos → equally weighted sources.

| Selector | Coverage (%) |
|---|---:|
| AutoGaze, forced fine K16 | 14.5064 |
| Center K16 | 36.1631 |
| Training-source prior K16 | 43.2043 |
| Human RL K16, six-seed mean | 44.3011 |

Primary RL minus source-prior difference: **+1.0968 pp**, paired source-stratified video-bootstrap 90% CI **[−0.6373, +2.9471]**. The primary result is **inconclusive**. RL exceeds the center control by +8.1381 pp [5.6537, 10.5403] and the forced fine-only AutoGaze control by +29.7947 pp [27.1824, 32.4437]. These are attention-coverage results, not QA results.

The validation advantage over the source prior (+2.82 pp in the prior study) does not establish a corresponding held-out advantage. The six-seed coverage SD is 0.0595 pp; this training-seed variation is separate from uncertainty across videos. The paired video intervals average the fixed seed set and are unadjusted.

Earlier no-KL policies and baselines were evaluated on this same test split. This is confirmation of the fixed 20k endpoints on historically exposed held-out data, not a pristine first test opening. Training, validation, and test video IDs are disjoint in the frozen manifest. The source prior was estimated exclusively from training mass, and its top-16 cells exactly match the original baseline.

Independent validation recomputed all seven action-export hashes, authoritative checkpoint bindings, per-video metrics, and all three paired bootstrap intervals. `test-validation.json` records the checks. The pretrained control differs from its historical aggregate by only 0.000379 pp; this comparison uses the newly evaluated control with the same current inference implementation.

The loading warning about `output_token_logit_bias` is a known historical-checkpoint compatibility case. CPU loading inspection verified an exactly zero persistent 266-entry buffer, the only missing key, for every model. Repeated initialization with different RNG seeds and zero-addition checks showed no logit or argmax changes. No inference was repeated for this audit; see `zero-bias-compatibility.json`.

`figures/fixed_k16_test_confirmation.*` shows coverage with six RL seed dots and the separate paired-video intervals. Model inference source: `94e3da6dee77ab6cdd3649e93dee5adf41ea30fe`; Slurm inference array 1990041; CPU curation 1990043. Heavy action traces remain outside Git.
