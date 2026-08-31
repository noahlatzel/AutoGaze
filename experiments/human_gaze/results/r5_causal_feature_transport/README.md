# R5 causal feature-transport logit bias

R5 tested whether gaze memory becomes useful when correspondence from the
previously selected connector features is added directly to the current
action logits through one zero-initialized scalar gate.

The fixed 10,000-update result is **inconclusive and practically null**.
All three paired endpoints improve (`+0.000826`, `+0.000714`, and
`+0.000381`), but their mean gain of `+0.000640` is far below the
preregistered `+0.005` effect. The matched-curve AUC mean is slightly negative
at `-0.000052`, and the descriptive mean improvement over R4 is only
`+0.000501` with a t-based 95% interval spanning zero.

The direct route is active and stable: the learned absolute gate reaches
roughly `0.0019`--`0.0025`, no source reaches the `-0.005` collapse boundary,
and every treatment remains strongly better than its learned static Top-16.
However, the proposed correspondence signal is poorly aligned with gaze. On
validation frames 1--15, its Top-16 achieves only `0.154276` macro-source
coverage, versus `0.442651` for carrying the previous coordinates forward and
`0.453087` for the learned policy. The correspondence prior is worse than
same-coordinate carry-over in both low- and high-motion strata.

The narrow conclusion is that shallow frozen-connector cosine similarity is
not a useful gaze-transport prior in this form. The result does not reject
explicit motion, learned correspondence, persistent memory, or longer-horizon
objectives; those are different interventions.

`metrics.json` contains the compact fixed comparison and transport-prior
diagnostic. `verification.json` records artifact/config invariants and confirms
that test remained unopened. Heavy checkpoints and per-clip diagnostics remain
under the paths in `manifest.json`; interpretation and figures live in
`wiki_gazing/Experiments/R5 Causal Feature Transport Logit Bias.md`.
