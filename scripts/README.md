# Scripts

This directory contains reusable helper scripts. Experiment identity, launch
grids, and run status belong under `experiments/`, not here.

## Layout

| directory | purpose |
| --- | --- |
| `runners/` | reusable executors for labels, features, probes, bandits, and HLVid evaluation |
| `analysis/` | plotting, inspection, and summarization utilities |
| `maintenance/` | artifact conversion and repair utilities |
| `examples/` | small training examples for the base AutoGaze package |

## Rules

- Put new reusable Python entrypoints in the matching subdirectory.
- Put experiment grids and run definitions in `experiments/<family>/configs/`.
- Use `experiments/tools/render_command.py` for local command rendering.
- Use `experiments/tools/submit_slurm.py` for new SLURM launches.
- Do not add new top-level `run_*.sh` or `*.sbatch` launchers here.

Historical launchers are archived in `experiments/archive/legacy_launchers/`.
Historical sbatch wrappers are archived in `experiments/slurm/legacy/`.

For HLVid proxy plots, use `scripts/analysis/plot_dry_proxy_sweep.py` for dry
proxy-sweep diagnostics and `scripts/analysis/plot_hlvid_proxy_benchmarks.py`
for completed NVILA benchmark frontiers.
