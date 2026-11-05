# Post-hoc controls for the fixed-K16 human-gaze confirmation (GT1) and HLVid answer flips

`evidential_status: post_hoc`. CPU-only re-analysis of already saved actions, cached human cell mass and saved HLVid predictions; zero model loads, inference or training. **The STAViS fold-1 test split was already opened at R0/R1 and again for GT1.** None of these analyses was preregistered; they are descriptive, unadjusted, and must not be used to select an endpoint or re-declare the GT1 decision (which remains *inconclusive*).

Reproduce from the AutoGaze root (refuses an existing output):
`OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= scripts/human_gaze/uv_autogaze_python.sh scripts/human_gaze/posthoc_gt1_controls.py --output <fresh>/<run_id> --readme <this README>`
(config `experiments/human_gaze/configs/gaze_posthoc_gt1_controls_20261105.yaml`, helpers `autogaze/human_gaze/posthoc_analysis.py`, test `tests/test_posthoc_analysis.py`). Loaders, coverage, aggregation (frames → clips → videos → equal-weight sources) and the paired source-stratified video bootstrap (10,000 iterations, seed 20261105, 90%) are the unchanged GT1 functions in `autogaze/human_gaze/supervised_analysis.py`. GT1 action-export manifests are checked against the hashes in the GT1 `execution_manifest.json`.

## 0. GT1 reproduction
The script asserts (tolerance 1e-12) that RL minus training-source prior reproduces GT1 exactly: **+1.0968 pp [−0.6373, +2.9471]**, and that the source-prior, Center16 and pretrained coverages match GT1 `metrics.json`.

## 1–2. Test coverage contrasts (six RL seeds, 67 videos)
| Comparator (test) | Coverage % | RL − comparator, source-macro pp [90% CI] | video-pooled pp [90% CI] |
|---|---:|---:|---:|
| Training-**source** prior16 (uses source label) | 43.20 | +1.10 [−0.64, +2.95] | +2.03 [−0.03, +4.27] |
| Training-**pooled** prior16 (source-agnostic) | 35.82 | +8.48 [+6.20, +10.74] | +5.40 [+3.34, +7.67] |
| Center16 | 36.16 | +8.14 [+5.65, +10.54] | +6.11 [+3.84, +8.55] |
| Learned static top16, validation-derived | 35.82 | +8.48 [+6.20, +10.74] | +5.40 [+3.34, +7.67] |
| Learned static top16, test-action-derived | 34.78 | +9.52 [+7.27, +11.72] | +6.35 [+4.30, +8.57] |
| Same-source shuffled video | 33.91 | +10.39 [+8.09, +12.84] | +9.76 [+7.36, +12.41] |
| Pretrained AutoGaze, forced fine K16 | 14.51 | +29.79 [+27.18, +32.44] | +25.37 [+22.20, +28.58] |

RL six-seed mean 44.30%. The source prior is a constant per-source map chosen from training attention mass **using the dataset/source label, which the policy never receives**; it is therefore privileged side information, not a fair label-free baseline. The pooled prior is the strongest label-free constant map tested; it shares 2–16 of its 16 cells with the per-source priors (Coutrot_db2: 2; DIEM: 16). The validation-derived static top16 of every seed is exactly the pooled-prior cell *set* (different order), hence identical numbers. Test-derived static uses each seed's own test selection frequencies (no human maps). Shuffle pairing: within each source, video IDs sorted; video *i* receives the RL actions of video *(i+1) mod n*; clip *j* receives clip *(j mod n_clips)* of that video; frames index-aligned (`same_source_shuffle_video_pairs` in `metrics.json`).

## 3. Action overlap on test (frame-level, source-macro, mean over seeds)
| Pair | intersection/16 | Jaccard |
|---|---:|---:|
| RL vs source prior16 | 58.5% | 44.3% |
| RL vs pooled prior16 | 61.0% | 47.6% |
| RL vs Center16 | 58.2% | 44.1% |
| RL vs pretrained fine K16 | 15.9% | 9.8% |
| RL seed vs RL seed (15 pairs) | 92.6% (92.0–93.2) | 87.0% (86.0–88.1) |
| Source prior vs pooled prior / Center16 | 72.9% / 66.7% | 63.8% / 53.6% |

Random 16-of-196 overlap would be 8.2%. The six seeds converge on nearly the same selections; roughly 40% of each frame's cells differ from any single constant map.

## 4. Validation K16 interval (427 clips, 34 videos)
Per-video RL values from the six step-20000 validation diagnostics (`outputs/human_gaze/diagnostics/r2b_fresh20k_stage3_*/center_collapse_val_step20000.json`); source prior recomputed from cell mass. Both reproduce R2c values (prior 42.378%, per-seed endpoints) within 1e-6. RL − source prior **+2.82 pp [+1.12, +4.65]** (video-pooled +4.48 [+2.23, +6.94]); RL − pooled prior +8.55 [+6.78, +10.34]. Validation shows a positive interval; test does not, but the test interval contains the validation point estimate.

## 5. Sensitivity: video-pooled weighting
Weighting each of the 67 test videos equally (instead of six equal sources) gives RL − source prior **+2.03 pp [−0.03, +4.27]**: larger but still touching zero. Source-macro weighting gives each of the 4 ETMD_av and 5 Coutrot_db2 videos far more weight than DIEM/Coutrot_db1 videos.

## 6. HLVid QA (268 questions, 77 videos)
Reused unchanged from `r2e_hlvid_k16_causal_controls/20260914-0149_r2e-k16-causal-curation_084d563/contrast_summary.csv` (copied as `hlvid_contrast_summary_reused.csv`):

| Trained − control | metric | diff pp | video bootstrap 90% | training-seed t 90% |
|---|---|---:|---:|---:|
| pretrained | question micro | +0.31 | [−2.32, +3.21] | [−0.59, +1.21] |
| pretrained | macro video | +1.74 | [−1.92, +5.42] | [+0.38, +3.10] |
| Center16 | question micro | −0.81 | [−3.37, +2.01] | [−1.71, +0.09] |
| Center16 | macro video | −1.97 | [−5.10, +0.85] | [−3.33, −0.61] |

Answer flips (`hlvid_answer_disagreement.csv`, `hlvid_disagreement_excess.csv`): mean per-question disagreement of the parsed answer letter is 8.3% between trained seeds (15 pairs; 5.97–10.82%), 12.1% trained vs Center16, 14.1% trained vs pretrained, 17.2% pretrained vs Center16. Across-minus-within excess (77-video cluster bootstrap): vs Center16 **+3.76 pp [+1.29, +6.28]**, vs pretrained **+5.75 pp [+3.17, +8.59]**; correctness flips +2.62 [+0.61, +4.64] and +6.11 [+3.63, +8.79]. 82.8% of questions receive the same letter from all six trained seeds. Selection therefore changes answers beyond seed noise, but the changes are close to net zero in accuracy. **Caveat:** trained arms ran on H100 NVL, controls on A40; part of the across-method excess may be cross-hardware numerical nondeterminism, which these data cannot separate (the two A40 controls also disagree on 17.2%).

## 7. Test-video count: 69 vs 67
D0 lists 69 test videos (`summary.json`, all present locally). Two AVAD test videos, **V29_Dog2 and V31_Bird, have only 15 target-grid frames (< 16-frame clip length) and therefore produce zero valid clips**; 67 source/video pairs remain in `clips.jsonl` and GT1. Separately, video names `clip4`, `clip14`, `clip15` occur in both Coutrot_db1 and Coutrot_db2 (64 unique names); all analyses key videos by (source, video_id), so these are not merged. See `test_video_reconciliation.csv`.

## Files
`metrics.json` (all results, prior/static cell lists, shuffle pairing), `coverage_by_method.csv`, `coverage_contrasts.csv`, `action_overlap_test.csv`, `hlvid_*.csv`, `test_video_reconciliation.csv`, `config.yaml`, `manifest.json` (sha256 of every input, script and module; heavy action exports stay at `/storage/user/latn/artifacts/gaze-completion/20261105-1447_gaze-completion_94e3da6/test_actions`, read only). Base code commit 0e11126.
