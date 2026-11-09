# N1 Native AutoGaze coverage

Run `20261108-1119_n1_native-autogaze-coverage_b48925b`. Gate note: `wiki/experiments/N1 Native AutoGaze Coverage.md` (wrapper).

Implementation check (pixel-union scorer reproduces GT1 fine-only values within 1e-06): pass.

Decision: **native_better**. Trained RL K16 (six-seed mean) minus Native-16, source-macro coverage: -0.2499 [-0.2785, -0.2211] (paired source-stratified video bootstrap, 90% interval, 10000 draws, seed 20261108).

| arm | coverage | area fraction | coverage / area | actions per frame |
|---|---|---|---|---|
| native16 | 0.6929 | 0.5543 | 1.250 | 16.00 |
| native_default | 0.6691 | 0.5571 | 1.201 | 50.07 |
| rl_six_seed_mean | 0.4430 | 0.0816 | 5.427 | 16.00 |
| center16 | 0.3616 | 0.0816 | 4.430 | 16.00 |
| pretrained | 0.1451 | 0.0816 | 1.777 | 16.00 |

Secondary contrasts (descriptive):

- trained_minus_native_default: -0.2261 [-0.2660, -0.1839]
- trained_minus_center16: +0.0814 [+0.0568, +0.1053]
- trained_minus_pretrained_fine16: +0.2979 [+0.2723, +0.3241]
- native16_minus_centre_at_area: -0.1463 [-0.1737, -0.1202]
- native_default_minus_centre_at_area: -0.0842 [-0.1088, -0.0602]

Coverage is the normalised attention mass inside the pixel union of the selected patches (2x2/4x4/7x7/14x14 patches are 112/56/32/16 px on the 224-px map), aggregated frames -> clips -> videos -> equal-weight sources. Centre@area: exact-area centred square with fractional boundary ring (see config.yaml).

Historically exposed test split (opened before GT1); prospective rule on an exposed population.

Files: metrics.json, config.yaml, manifest.json, coverage_by_source.csv, arm_summary.csv.
