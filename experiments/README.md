# AutoGaze Experiment Registry

This directory is the coordination layer for research experiments around
temporal proxy metrics, delta policies, and HLVid/NVILA evaluation. The
`autogaze/` package remains the implementation layer; experiment definitions,
launch metadata, run ledgers, and result summaries live here.

## Experiment Families

| family | purpose | active files |
| --- | --- | --- |
| `proxy_sampling` | HLVid frame/tile selection with thumbnail and semantic proxy metrics | `proxy_sampling/configs/`, `proxy_sampling/manifests/` |
| `delta_policy` | pseudo-label generation, frozen AutoGaze feature probes, contextual-bandit delta policies | `delta_policy/configs/`, `delta_policy/manifests/` |
| `hlvid_nvila` | NVILA-HD-Video baselines and fixed-delta evaluations | `hlvid_nvila/configs/`, `hlvid_nvila/manifests/` |

## Status Vocabulary

Use these statuses in configs and manifests:

| status | meaning |
| --- | --- |
| `planned` | defined, not launched |
| `queued` | submitted to SLURM, not complete |
| `running` | currently executing |
| `completed` | result artifact exists and has been inspected |
| `failed` | launched but did not produce usable output |
| `superseded` | kept for provenance but not a current decision point |

## Run Definition Contract

Every run config should make the following explicit:

- `id`: stable identifier used in output paths and summaries.
- `family`: one of the experiment families above.
- `status`: lifecycle state.
- `question`: research question the run answers.
- `runner`: script or module that executes the run.
- `environment`: conda environment or runtime requirement.
- `dataset`: logical dataset name and path if machine-specific.
- `outputs`: expected output directory and key result files.
- `command`: structured arguments needed to reproduce the run.
- `metrics`: primary and secondary measurements.
- `notes`: related research notes.

Keep run configs small and specific. Put grids in a family config when all runs
share the same question and execution path.

## Output Convention

Prefer:

```text
outputs/<family>/<experiment_id>/
  run_manifest.yaml
  command.sh
  results.jsonl
  summary.json
```

Existing historical outputs are not moved. They are referenced from manifests so
past results remain discoverable without changing provenance.

## Current Priorities

1. Register existing proxy-sampling and delta-policy results.
2. Move future sweeps from Python constants and shell env vars into YAML configs.
3. Route local and SLURM launches through the small launch helpers in each
   experiment family.
4. Keep AutoGaze model, dataset, and training code changes separate from
   experiment bookkeeping.
