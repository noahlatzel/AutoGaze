# Figure captions

`actual_vs_forced_k16`: HLVid video-macro exact-match accuracy for the same three
independently trained R2d checkpoints under calibrated variable EOS and coherent
forced K16. Each endpoint contains all 268 official test questions / 77 videos.
Lines join seeds only for legibility, not training trajectories. Costs are not
shown in accuracy-only curation because full variable observations are missing.

`paired_accuracy_deltas`: Within-checkpoint variable-EOS minus forced-K16 accuracy
(percentage points); video-macro is primary and question-micro is sensitivity.
Gray dots are the three paired seeds. Blue intervals are paired video-cluster
90% percentiles (10,000 resamples jointly shared across policies and seeds);
orange intervals are t95 intervals over the three independent seed contrasts.
These separate uncertainty views are not combined. Deployment EOS also changes
later spatial ordering: this is not an allocation-only causal comparison.
