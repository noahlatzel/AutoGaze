# B0-B3 fine-grid human coverage baselines

Primary aggregation is mean over clips within video, mean over videos within
source, then an unweighted mean over the six sources. Prior ranks cells from
training videos of the same source only. Random is averaged over five fixed
seeds. The full per-video result is in `metrics.json`.

## Macro-source coverage

### Validation

| K | Random | Center | Prior | Oracle |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0053 | 0.0313 | 0.0486 | 0.1649 |
| 2 | 0.0102 | 0.0627 | 0.0806 | 0.2883 |
| 4 | 0.0201 | 0.1161 | 0.1530 | 0.4569 |
| 8 | 0.0410 | 0.2104 | 0.2779 | 0.6563 |
| 16 | 0.0810 | 0.3689 | 0.4238 | 0.8495 |

### Test

| K | Random | Center | Prior | Oracle |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0051 | 0.0283 | 0.0439 | 0.1626 |
| 2 | 0.0099 | 0.0582 | 0.0754 | 0.2887 |
| 4 | 0.0199 | 0.1088 | 0.1461 | 0.4587 |
| 8 | 0.0412 | 0.1989 | 0.2491 | 0.6574 |
| 16 | 0.0815 | 0.3616 | 0.4320 | 0.8500 |

At K=16, the train-only source prior beats center in the six-source macro but
not in every source. The test prior is especially strong for Coutrot_db2
(0.6155) and ETMD_av (0.5095), while Center-16 is slightly stronger for AVAD
and weaker sources remain close. Oracle-16 retains 0.8500 macro coverage,
leaving substantial content-dependent headroom over the fixed prior.

## Command

```bash
conda run -n autogaze python scripts/human_gaze/evaluate_coverage_baselines.py \
  --config experiments/human_gaze/configs/b0_coverage_baselines_fold1.yaml
```
