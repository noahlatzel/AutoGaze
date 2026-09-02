# Exact-K32 fixed-budget reproduction

This bundle reproduces the completed R2c exact-K36 protocol at exact K32. The only scientific treatment change is the fixed fine-cell budget (36 to 32); run-identity paths and reporting prefixes change accordingly. All three matched seeds use the fixed 20,000-update endpoint on the unchanged train-derived validation population. The protected test split was not opened.

The learned K32 endpoint is `0.632230 ± 0.000942` (population SD) versus Prior-32 `0.640450`, a delta of `-0.008220`. The frozen coverage rule is **fail**. The endpoint dynamic/static/shuffled content-dependence rule is **pass**.

`metrics.json` contains per-seed endpoints, K16/K24/K32/K36 comparisons, per-source K32 estimates, and the complete longitudinal control suite. `verification.json` proves exact 32-per-frame / 512-per-clip accounting, all 20,000 updates, fixed validation schedules, matched seeds, and resolved-config parity with K36 except the budget and run identity. `manifest.json` records commits, jobs, hashes, compute, and heavy-artifact paths. `figures/` contains the four summary plots plus the nine seed-by-checkpoint concentration heatmaps used by the diagnostics.
