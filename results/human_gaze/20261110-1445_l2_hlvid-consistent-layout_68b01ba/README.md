# L2 gaze HLVid rerun with a consistent processor layout: `20261110-1445_l2_hlvid-consistent-layout_68b01ba`

The processor addresses patches in the vision tower's layout, so the cells a policy selects are the
patches the answerer encodes. Question-micro accuracy on 268 HLVid questions (77 videos), 16 fine cells
per 224 px sub-tile and frame, 128 uniform frames, 64 thumbnails; 90% paired video bootstrap.

**Floor (Center16 minus Random16): +2.24 pp [-0.45, +4.74] — unresolved (no equivalence claim).**

**Gaze (two trained seeds minus Center16): +0.00 pp [-3.08, +2.75] — unresolved (no equivalence claim).**

| Arm | Micro (%) | Macro (%) | Cells in central block (%) | GPU |
|---|---|---|---|---|
| center16 | 48.51 | 48.16 | 100.0 | NVIDIA A40 |
| random16 | 46.27 | 47.74 | 8.2 | NVIDIA A40 |
| pretrained | 51.87 | 51.13 | 0.8 | NVIDIA A40 |
| trained_440826 | 48.13 | 46.18 | 62.7 | NVIDIA A40 |
| trained_440827 | 48.88 | 46.58 | 63.6 | NVIDIA A40 |

| Arm | Released layout, micro (%) | Consistent minus released (micro) | Letters changed (%) |
|---|---|---|---|
| center16 | 51.12 | -2.61 pp [-6.67, +1.39] | 20.1 |
| random16 | 44.78 | +1.49 pp [-1.46, +4.69] | 11.2 |
| pretrained | 50.00 | +1.87 pp [-1.55, +5.77] | 17.2 |
| trained_440826 | 51.12 | -2.99 pp [-6.59, +0.40] | 19.8 |
| trained_440827 | 49.25 | -0.37 pp [-4.74, +3.53] | 21.6 |

The released-layout rows come from other jobs and partly other hardware and are descriptive.
All arms on one GPU model: True. Exposed population; the gaze interval is conditional on two seeds.
Protocol and decision rules: `config.yaml`. Heavy outputs: `manifest.json`.
