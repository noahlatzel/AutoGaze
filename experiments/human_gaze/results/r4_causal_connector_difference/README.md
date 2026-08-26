# R4 causal connector feature difference

R4 tested whether a framewise normalized connector-feature difference, added
through one zero-initialized scalar gate, improves exact-K16 human-gaze
coverage over the immutable R3 controls.

The fixed 10,000-update result is **inconclusive and practically null**. Paired
endpoint changes are `+0.001413`, `-0.000269`, and `-0.000725`; their mean is
`+0.000140` with a t-based 95% interval of `[-0.002658, +0.002937]`. The
matched-curve AUC mean is `+0.000020`. No source mean falls below the
preregistered `-0.005` collapse boundary, and every treatment remains strongly
better than its learned static Top-16.

The treatment is stable, but its gate stays tiny: the maximum logged
signal/feature-RMS ratios are `0.001956`, `0.001746`, and `0.001618`. This
supports only the narrow conclusion that an additive local-difference residual
does not provide a useful route into this decoder; it does not rule out motion
or gaze memory applied through a more direct mechanism.

`metrics.json` contains the fixed comparison and `verification.json` records
artifact/config invariants and confirms that test remained unopened. Heavy
per-clip history and center-control outputs remain under the paths in
`manifest.json`; interpretation and figures live in
`wiki_gazing/Experiments/R4 Causal Connector Feature Difference.md`.
