# F1 Gaze HLVid selection floor — `20261106-1052_f1_hlvid-selection-floor_98c0dce`

Question: is HLVid accuracy (frozen NVILA-8B-HD-Video, 128 frames, 64 full
thumbnails, max_tiles_video=48, exact fine-only K16, greedy) sensitive to
within-frame selection? Arms used frozen commit `98c0dce3d7e33dbe0ad982d4fda1ce7e4f49e656` on RTX PRO 6000 Blackwell GPUs. Random16 completed in job1992333; Center16 retained its first nine answers and resumed alone in an automatically requeued recovery allocation:
Random16 (seed 20261106, per video/tile/frame uniform
draw at the Center16 boundary) and an unchanged Center16 rerun. Thumbnails-only
was not run (see metrics.json). 268 questions / 77 videos per arm.

Question-micro accuracy: Random16 44.78%, Center16 51.12%.
Primary Center16 − Random16: +6.34 pp, 90% paired
77-video bootstrap [+2.97, +9.92] pp
(10,000 draws, seed 20261107). Decision: sensitive: Center16 above the Random16 floor (90% lower bound > 0).

Secondary rows in `contrasts.csv` (macro accuracy; trained six-seed mean −
Random16 is cross-job/cross-hardware and descriptive only). Answer disagreement
rates and the Center16 rerun-vs-R2e reproduction check are in metrics.json.
Per-question letters, per-video accuracy and per-example telemetry are included.
Heavy evidence (events, evidence records, logs) stays at `/storage/user/latn/artifacts/f1/20261106-1052_f1_hlvid-selection-floor_98c0dce`.
Reproduce: `scripts/human_gaze/curate_f1_selection_floor.py --config
experiments/human_gaze/configs/f1_hlvid_selection_floor_v1.yaml --raw-root <raw> --run-id 20261106-1052_f1_hlvid-selection-floor_98c0dce
--expected-commit 98c0dce3d7e33dbe0ad982d4fda1ce7e4f49e656 --output-dir <fresh>`.

Recovery: the user approved an eight-hour cumulative Center16 recovery budget on 2026-11-07, beyond the failed allocation (23,908 seconds). Allocation receipts and logs: `/storage/user/latn/artifacts/f1/20261106-1052_f1_hlvid-selection-floor_98c0dce/recovery/20261107-oom`.
