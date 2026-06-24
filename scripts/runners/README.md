# Runners

These are reusable execution entrypoints. Experiment identity and run status
live under `experiments/`.

## HLVid Runner Responsibilities

| runner | responsibility | experiment family |
| --- | --- | --- |
| `run_hlvid_proxy_dry_sweep.py` | local, in-process proxy grid sweep without NVILA generation; measures selected deltas, visited-frame counts, and proxy scoring cost | `proxy_sampling` dry sweeps |
| `evaluate_hlvid_proxy_sampling.py` | full HLVid/NVILA evaluation for uniform or proxy-guided sampling, with detailed timing and optional single-run dry sampling | `proxy_sampling` full NVILA benchmarks |
| `evaluate_hlvid_nvila.py` | direct NVILA baseline/control evaluation for default AutoGaze, fixed inter-tile deltas, and dense/fixed controls | `hlvid_nvila` baselines |

Keep both HLVid evaluators for now. `evaluate_hlvid_nvila.py` is the small
baseline/control runner. `evaluate_hlvid_proxy_sampling.py` is the heavier
proxy-policy runner with proxy scoring, cost selection, and detailed timing.
Shared HLVid utilities belong in `hlvid_common.py`.
