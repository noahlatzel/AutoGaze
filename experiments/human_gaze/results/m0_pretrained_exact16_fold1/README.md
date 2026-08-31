# M0 pretrained AutoGaze fine-only exact-16

The local pretrained checkpoint was evaluated without weight updates. Actions
0-68 and EOS were masked, generation was greedy, repeats were prohibited, and
every frame produced exactly 16 actions from IDs 69-264. Global action IDs were
converted to 0-195 fine-grid cells before coverage evaluation.

| Split | Macro-source coverage | Pooled-frame coverage | Clips |
| --- | ---: | ---: | ---: |
| Validation | 0.1191 | 0.1099 | 427 |
| Test | 0.1451 | 0.1145 | 663 |

## Test coverage by source

| Source | Coverage |
| --- | ---: |
| AVAD | 0.2347 |
| Coutrot_db1 | 0.1606 |
| Coutrot_db2 | 0.1182 |
| DIEM | 0.1444 |
| ETMD_av | 0.1006 |
| SumMe | 0.1118 |

The pretrained policy exceeds Random-16 but is far below Center-16 and Prior-16.
The first full attempt at batch size 4 was stopped before producing output after
GPU monitoring showed substantial unused memory. The completed deterministic
run used batch size 16 on the local RTX A2000 12 GB; changing inference batch
size does not change selected actions.

```bash
HF_HOME=/storage/slurm/latn/data/huggingface_cache \
conda run -n autogaze python scripts/human_gaze/evaluate_pretrained_exact16.py \
  --config experiments/human_gaze/configs/m0_pretrained_exact16_fold1.yaml
```
