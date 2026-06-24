# Dataset and Artifact Locations

These paths are machine-specific and should not be assumed portable. Keep them
as documented defaults or config values, not hard-coded into AutoGaze internals.

## AutoGaze Training Data

```text
/storage/slurm/latn/data/AutoGaze/AutoGaze-Training-Data/
  InternVid_res448_250K/
  100DoH_res448_250K/
  Ego4D_res448_250K/
  scanning_SAM_res448_50K/
  scanning_idl_res448_50K/
  gazing_labels.json
```

## HLVid

```text
/storage/slurm/latn/data/AutoGaze/HLVid/
  data/test-00000-of-00001.parquet
  videos/
  videos_part_0001.tar ... videos_part_0016.tar
```

## Model Mirrors

```text
/storage/slurm/latn/data/AutoGaze/hf_models/
  facebook--vit-mae-large/
  facebook--dinov2-with-registers-base/
  google--siglip2-base-patch16-224/
  nvidia--AutoGaze/
  nvidia--NVILA-8B-HD-Video/
```

Frozen VideoMAE reconstruction checkpoint:

```text
/storage/slurm/latn/data/AutoGaze/VideoMAE_AutoGaze/videomae.pt
```

Recommended Hugging Face cache:

```text
HF_HOME=/storage/slurm/latn/data/huggingface_cache
```
