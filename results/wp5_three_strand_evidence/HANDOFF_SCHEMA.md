# Study-owner handoff schema

Provide readable prose plus: source table and result-bundle paths; observational unit and aggregation order; primary metric, direction, and units; sensitivity metrics; exact split and historical exposure; method and checkpoint identifiers and hashes; repository commit and dirty-worktree flag; data manifest and hash; all valid seeds or repeats and reasons for exclusions; uncertainty unit and construction; compute boundary and selection overhead; evidence status; limitations; SVG/PDF/PNG paths; and an exact regeneration command.

Repeated frames or deterministic encoder calls are not training seeds. Paired video-level uncertainty does not replace training-seed uncertainty. Analytical operation counts are not measured latency. Processor replay is not end-to-end generation timing.

Token-merging fidelity is encoder-level only. Temporal selection does not establish spatial efficiency. K16 gaze causal comparisons use all six valid endpoints; K16/K24/K32 budget comparisons use the three shared seeds.

CSV seed identifiers carry a `seed_id_role`. Base IDs used to match experiments
must be distinguished from the training RNG seed; preserve their mapping in
provenance. Deterministic controls have inapplicable, empty `n_seeds`, not one
training seed. State whether a mean action budget averages sources equally or
pools frames before interpreting it as a resource measure.

The normalized metric table additionally supports `metric_unit`,
`aggregation_unit`, `n_questions`, `n_videos`, `bootstrap_seed`, `source_field`,
`metric_role`, and `protocol_validity`. Blank primary designation means the
historical source did not declare a primary metric; a figure's chosen reporting
axis must not be relabeled as preregistration. Bootstrap RNG identities belong
in `bootstrap_seed`, with training-seed counts left unavailable. Accuracy
fractions and fraction differences become percent or percentage points only
when a figure explicitly applies a factor of 100.

Pinned external figure links are permitted. Their regeneration command is run
in the linked source repository at the recorded commit, into a new output
directory. Addendum manifests retain exact source hashes and name superseded
bundles without altering completed source artifacts.

DINO rows add `n_sources`, `n_frames_per_video`, `replicate_kind`,
`replicate_count`, and `timing_protocol`. Human endpoint counts may populate
`n_seeds`; random-mask replication belongs in the distinct replicate fields and
has empty training-seed count. Preserve base-ID to continuation-training-seed
mapping separately. Frame counts and bootstrap draws are not additional training
replicates. Deterministic analytical FLOP counts have no uncertainty interval.
Batch-16 amortized milliseconds/frame and the separate two-video batch-one
milliseconds/frame share a physical unit but differ in execution and sampling.
