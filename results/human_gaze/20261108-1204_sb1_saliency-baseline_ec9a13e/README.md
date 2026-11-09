# SB1 Saliency model baseline (TASED-Net, DHF1K)

Run `20261108-1204_sb1_saliency-baseline_ec9a13e`. Gate note: `wiki/experiments/SB1 Saliency Model Baseline.md` (wrapper).

Implementation check (GT1 trained six-seed mean, Center16, train source prior within 1e-06): pass.

Decision: **saliency_better**. Trained RL K16 (six-seed mean) minus Saliency-top16, source-macro coverage: -0.1174 [-0.1416, -0.0939] (paired source-stratified video bootstrap, 90% interval, 10000 draws, seed 20261109).

| arm | coverage (source-macro) |
|---|---|
| saliency_top16 | 0.5604 |
| rl_six_seed_mean | 0.4430 |
| best_constant | 0.3635 |
| train_source_prior | 0.4320 |
| center16 | 0.3616 |

Secondary contrasts (descriptive):

- saliency_minus_center16: +0.1988 [+0.1664, +0.2306]
- saliency_minus_train_source_prior: +0.1284 [+0.0992, +0.1578]
- trained_minus_best_constant: +0.0795 [+0.0554, +0.1028]

Cell overlap |S ∩ X| / 16 (source-macro): saliency vs trained (seed mean) 0.508; saliency vs Center16 0.450; trained vs Center16 0.582.

Saliency-top16: TASED-Net (MichiganCOG/TASED-Net @ 0cf47b7, TASED_updated.pt, trained on DHF1K with a Kinetics-400 S3D encoder; no STAViS source) with its run_example.py protocol (32-frame window of native frames ending at each target frame, 224x384 input, 8-bit + Gaussian sigma 7 postprocessing), map resized to 224x224 (PIL bilinear), top-16 of 14x14 cell masses, ties by cell index. Best constant: 16 cells with the largest training attention mass under the evaluation weights. See config.yaml.

Historically exposed test split (opened before GT1); prospective rule on an exposed population.

Files: metrics.json, config.yaml, manifest.json, coverage_by_source.csv.
