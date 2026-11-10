# Frame Selector Experiments

This family tracks external fixed-budget frame-selection baselines for HLVid.
Selectors generate frame-index JSONL files that are later consumed by
`scripts/runners/evaluate_hlvid_nvila.py --frame-source selected_indices`.

`infoshot_reimpl` is an independent implementation from the published paper;
the public upstream repository contained no implementation code at integration
time. It is intentionally named differently from an official reproduction.

Primary interface:

```text
video + question + K=128 -> selected source-frame indices
```

## Model Registry

Model choices for external selectors are recorded in `models.yaml`. The scripts
use the `local_path` values as defaults, while still allowing command-line
overrides such as `--clip-model` or `--mif-sim-model`.

Durable model mirrors live under:

```text
/storage/slurm/latn/data/AutoGaze/hf_models/
```

Use the shared Hugging Face cache for downloads and cluster jobs:

```text
HF_HOME=/storage/slurm/latn/data/huggingface_cache
```
