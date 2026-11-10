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
| `frame_selectors` | External and paper-reimplemented fixed-budget HLVid frame selectors | `frame_selectors/configs/`, `frame_selectors/models.yaml` |

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

## Configs vs Manifests

Configs are the source of runnable intent: they describe the runner, command
arguments, grids, and per-run overrides. A run with `launchable: false` is a
historical or aggregate entry and should not be rendered as a command.

Manifests are the source of observed state: they record completed, planned,
queued, failed, or superseded runs and point at result artifacts. The summary
tool reads manifests because it is reporting what exists, not every possible
config expansion.

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
3. Route local and SLURM launches through `experiments/tools/`. Family launch
   helpers are thin aliases for convenience.
4. Keep AutoGaze model, dataset, and training code changes separate from
   experiment bookkeeping.
