# R2d variable fine-token budget

R2d tested whether a learned EOS policy can allocate 4--36 fine spatial tokens
per frame more usefully than fixed K16. It initialized from the three completed
R2b K16 endpoints, retained the human-coverage reward and frozen visual stack,
and trained a dual-controlled token price for 20,000 updates. A one-dimensional
EOS bias was recalibrated on the preregistered source-balanced training subset
before the fixed validation endpoint.

The final gate is **negative; retain fixed K16**. Calibrated validation mean K
is `15.073`, `15.038`, and `15.357` (mean `15.156`), below the matched-compute
band `[15.5, 16.5]`. Variable-policy macro-source coverage is `0.37710`,
`0.37810`, and `0.38106`, while forcing the same trained endpoints to exactly
K16 gives `0.41374`, `0.41754`, and `0.42007`. The paired mean difference is
`-0.03837` with a descriptive t-based 95% interval
`[-0.04210, -0.03463]`.

The isolated allocation control is positive: applying actual versus
same-source shuffled lengths to the same forced-K36 spatial ordering improves
coverage by `+0.00772`, `+0.00672`, and `+0.00885`. Thus the policy learns a
small reproducible content-dependent length signal, but joint training damages
the spatial policy and the selected lengths do not produce a competitive
combined policy. Forced-K16 coverage itself falls `-0.03453` on average from
the matched R2b initialization.

The calibrated length distribution remains broad (`10.37` frame-level standard
deviation), with mean minimum/cap rates `0.113`/`0.051`. Length correlates only
weakly with gaze entropy (`0.012`) and center mass (`0.077`). Per-source mean K
ranges from `11.38` on SumMe to `19.30` on AVAD, and variable coverage is below
forced K16 on five of six sources.

`metrics.json` contains the compact fixed-endpoint comparison.
`verification.json` records run/config/data invariants and confirms validation
only. Heavy checkpoints, histories, and per-clip diagnostics remain at the
paths in `manifest.json`; interpretation and figures live in
`wiki_gazing/Experiments/R2d Variable Fine-Token Budget.md`.
