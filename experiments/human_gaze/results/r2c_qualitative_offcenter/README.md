# R2c qualitative off-center examples

This result selects three distinct-source validation clips using only ground-truth gaze mass outside Center-32, then renders four consecutive 3 Hz samples with ground-truth heatmaps and learned K16, K24, and K32 gaze cells.

`qualitative_manifest.json` records the selection rule, checkpoints, clips, frame numbers, selected cells, and per-frame coverage. The K32 cells are exact-32 inference from the K36-trained decoder, not a separately trained K32 model.

The rendered figures live in the wrapper vault at `wiki_gazing/assets/r2c_qualitative_offcenter/` and are interpreted in `wiki_gazing/Experiments/R2c Qualitative Off-Center Examples.md`.

Regenerate from the AutoGaze repository root with:

```bash
HF_HOME=/storage/slurm/latn/data/huggingface_cache conda run --no-capture-output -n autogaze \
  python scripts/human_gaze/visualize_fixed_budget_qualitative.py \
  --root /storage/user/zverev/datasets/av-gaze-stavis \
  --manifest outputs/human_gaze/d0_stavis_fold1_validated/clips.jsonl \
  --cell-mass outputs/human_gaze/d0_stavis_fold1_validated/cell_mass.npy \
  --k16-checkpoint outputs/human_gaze/grpo/r2b_fresh20k_stage3_base440826_seed540826/checkpoint_latest_gaze \
  --k24-checkpoint outputs/human_gaze/grpo/r2c_fixedk24_stage3_base440826_seed540826/checkpoint_latest_gaze \
  --k32-checkpoint outputs/human_gaze/grpo/r2c_fixedk36_stage3_base440826_seed540826/checkpoint_latest_gaze \
  --split val --num-clips 3 --frames-per-clip 4 \
  --output-dir ../wiki_gazing/assets/r2c_qualitative_offcenter \
  --output-manifest experiments/human_gaze/results/r2c_qualitative_offcenter/qualitative_manifest.json
```
