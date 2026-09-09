# Temporal frame-selection evidence closure

Run `20260909-0315_t0-historical-temporal-evidence_50c1949` reaggregates existing raw artifacts only: **zero new VLM calls and zero new GPU runs**.

## Supported claim
Some historical proxy settings retained similar observed HLVid QA while keeping fewer candidate-grid frames. This is bounded descriptive evidence, not equal-budget superiority, a confirmed threshold, a controlled speedup, or an integrated three-method acceleration.

HLVid has 268 questions over 77 videos (146 audiovisual, 122 household). Historical evaluation used NVILA-8B-HD-Video, prompt `<video>\n\nQuestion: {question}`, up to 16 generated tokens, and the first standalone A–D letter. Exact NVILA/AutoGaze revisions were not embedded in the early bundles and remain missing.

## Core evidence
| Condition | Max tiles/video | Micro | Video macro (90% CI) | Frames | Tokens | Total/proxy s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform 128 | 48 | 48.51% | 50.10% (42.94, 57.39) | 128.0 | 26363 | 169.4/0.0 |
| DINOv2 novelty | 48 | 48.51% | 51.07% (44.00, 57.99) | 58.3 | 14673 | 68.1/12.6 |
| VideoMAE reconstruction | 48 | 45.90% | 46.02% (39.02, 53.04) | 60.4 | 14733 | 61.6/7.6 |
| Uniform 256 | 48 | 50.00% | 51.79% (45.16, 58.23) | 256.0 | 44405 | 221.1/0.0 |
| Pixel MSE t=.04 | 48 | 52.99% | 53.72% (46.69, 60.83) | 211.5 | 37885 | 206.4/24.9 |
| SigLIP perceptual | 48 | 49.63% | 50.29% (43.46, 57.08) | 239.5 | 42016 | 270.4/10.0 |
| Learned DINO-delta policy | 1 | 44.03% | 43.46% (37.25, 49.97) | 48.0 | missing | missing |

Micro weights questions; macro weights videos equally. Frames, tokens and time are averaged over questions. The 128/256 values are candidate-grid sizes, not raw source frames. `conditions.csv` retains source-frame ranges. Threshold sweeps are exploratory because the HLVid test was repeatedly exposed. Timing is descriptive: cold-decode and frame-cache-hit question counts vary by condition (see `timing_cache_strata.csv`), query-free proxy scores were recomputed per question, and DINO diversity used different hardware.

## Learned delta
Delta is a gap after a 16-frame kept block, so gap 32 yields starts `(0,48,96)`. The learned target maximizes future DINOv2 novelty over gaps {1,2,4,8,16,32}; future features are noncausal teacher labels. Training used consecutive short clips, while deployment used a sparse full-duration 128-point grid and passed `frame_count=128` plus `effective_fps=128/duration`.

All 77 videos realized one schedule: starts `(0,48,96)`, actions `(32,32,16)`. This fixed-schedule collapse is the strongest negative result. The learned QA rollout used `max_tiles_video=1`, while the uniform and fixed-gap references used `max_tiles_video=48`. Its QA difference is therefore resolution-confounded, including against fixed gap 32, despite that condition realizing the same frame schedule. No same-budget, matched-resolution uniform-48 QA run exists. The collapse conclusion follows from the trajectories and does not require a QA comparison.

## Reconciled boundaries
T0d is temporal within-tile promotion, not HLVid QA or decode avoidance. Its protected-test oracle improved relative distortion 15.03%; reversible selection was 8.29% worse than uniform and 8.16% worse than forward-only. These are relative distortion changes, not QA points. Its manifest records execution at AutoGaze `a1bfdb5100bd89a5d6f72fccc95627b662548b58` and wrapper `cbd30290c952bb436d946262969e223b76a28eeb`. Immediate curation used the child AutoGaze commit `8e0a176` and wrapper `0db19a9`; these are compatible execution/curation roles.

P0 is spatial gazing. Prior+SURGE lifted coverage 0.0109 (95% CI 0.0051, 0.0166), but raw SURGE lost to RGB and feature-difference controls, so its gate failed. Its verified bundle is indexed in `supplemental_inventory.csv` for the gazing evidence owner.

## Files
`conditions.csv`, `paired_comparisons.csv`, `answer_flips.csv`, and `timing_cache_strata.csv` are normalized source tables. `learned_trajectories.csv` preserves every actual rollout. `method_inventory.csv` maps mechanism and information to artifacts, limitations, and claims. `supplemental_inventory.csv` carries learned-probe/T0d/P0 evidence. `figures/` contains PDF/SVG/PNG triplets and `figure_captions.md` contains bounded captions. `manifest.json` records hashes, sizes, UTC timestamps, and provenance gaps.

## Remaining work
No new run is needed for the bounded claim. Uniform-48 QA at `max_tiles_video=1` would be new optional evidence only if the thesis later demands a matched-budget, matched-resolution learned-policy comparison. The collapse result needs no repeated QA.
