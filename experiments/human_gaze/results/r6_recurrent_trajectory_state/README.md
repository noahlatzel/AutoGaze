# R6 learned recurrent trajectory state

R6 isolated a learned, selection-conditioned recurrent policy state while
holding the 16-frame horizon, exact K16 budget, reward, frozen visual stack,
decoder, sampling, and continuation protocol fixed. The state is a 192-wide
GRU updated once after each completed frame from the mean selected frozen
connector representation. A zero-initialized scalar gate exposes a normalized
state readout as a 196-cell action-logit residual, giving exact control identity
at initialization.

The fixed 10,000-update gate is **inconclusive and practically null**. Paired
treatment-minus-R3-control endpoint effects are `+0.000892`, `+0.000311`, and
`-0.001621`; the mean is `-0.000139` (descriptive t-based 95% interval
`[-0.003408, +0.003129]`). Only one pair is nonpositive, so the preregistered
fail rule requiring two nonpositive pairs is not met, but the result is far
below the `+0.005` pass threshold. Matched-curve AUC is effectively zero
(`+0.000005`). No source has a mean loss of `-0.005`, and effects are mixed
across sources and seeds.

The pathway is stable and technically active. Endpoint absolute gates are
`0.0040`--`0.0058`; normal-versus-reset state-only selection Jaccard is
`0.99094`, below the `0.999` inert criterion; maximum state RMS is `0.50226`;
and no state component reaches the saturation threshold. Dynamic validation
coverage remains above Center-16, learned static Top-16, and shuffled-video
controls in every seed.

The trajectory diagnostics do not support improved target following. Adjacent
selection overlap rises only `+0.00269` over control, so there is no blind-copy
collapse, but policy centroid displacement falls by `0.00393` cells/frame.
Overall centroid-velocity error improves by only `0.00246` cells and abrupt
transition error worsens by `0.00046` cells. State reset and alternative-state
interventions make almost no coverage difference. The learned policy remains
better than same-coordinate carry-over in both motion strata, but this is not
an isolated recurrence benefit.

The narrow conclusion is that this simple explicit persistent state is
trainable but does not repair trajectory undertracking under the current
reward and frozen representation. It should not be adopted. A higher-value
next gate is a cheap train-to-validation causal trajectory probe that tests
whether frozen connector history contains learnable next-gaze displacement
information before another policy architecture or objective is changed.

`metrics.json` contains the compact fixed comparison and trajectory
diagnostics. `verification.json` records training, stability, gradient, and
split invariants. Heavy artifacts remain at the paths in `manifest.json`;
interpretation and figures live in
`wiki_gazing/Experiments/R6 Learned Recurrent Trajectory State.md`.
