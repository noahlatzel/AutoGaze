# D0 AV-gaze-STAViS alignment audit

## Configuration

- Fold: 1 for AVAD, Coutrot_db1, Coutrot_db2, ETMD_av, and SumMe; supplied
  DIEM split
- Target rate: 3 fps, nearest native frame from a grid beginning at t=0
- Clip: 16 complete frames, stride 16 target frames
- Spatial transform: direct full-field bilinear resize to 224 x 224
- Validation: deterministic 20% of official training videos per source, seed
  140826
- Target: supplied blurred map normalized per frame and summed into 14 x 14
  fine-grid cell masses

## Result

The initial path/existence manifest contained 2,953 clips. Full heatmap
validation excluded nine clips with at least one zero-mass map: eight DIEM and
one SumMe. The final manifest contains 2,944 clips and 47,104 frames:

| Split | Videos | Valid clips |
| --- | ---: | ---: |
| Train | 135 | 1,854 |
| Validation | 34 | 427 |
| Test | 69 | 663 |

The cached `float32` cell-mass tensor has shape `[2944, 16, 196]`; per-frame
sums range from 0.99999982 to 1.00000012. All source datasets occur in every
split. No video crosses split boundaries.

Heavy derived files are ignored under
`outputs/human_gaze/d0_stavis_fold1_validated/`. The directory contains
`clips.jsonl`, `summary.json`, `invalid_clips.json`, `cell_mass.npy`, and the
resolved target-cache config. Source datasets were only read.

## Commands

```bash
conda run -n autogaze python scripts/human_gaze/build_stavis_manifest.py \
  --config experiments/human_gaze/configs/d0_stavis_fold1.yaml

conda run -n autogaze python scripts/human_gaze/cache_stavis_cell_mass.py \
  --config experiments/human_gaze/configs/d0_stavis_targets_fold1.yaml

conda run -n autogaze python scripts/human_gaze/validate_stavis_manifest.py \
  --root /storage/user/zverev/datasets/av-gaze-stavis \
  --manifest outputs/human_gaze/d0_stavis_fold1/clips.jsonl \
  --max-clips-per-source 1
```
