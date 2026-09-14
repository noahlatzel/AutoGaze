# R2e exact-K16 HLVid controls: completed descriptive comparison

Run bundle `20260914-0149_r2e-k16-causal-curation_084d563`. Official HLVid test: 268 questions / 77 videos.
Pretrained and tile-local Center16 controls complete 536 unique QA. Six trained
20k K16 endpoints are reused unchanged (1,608 QA); zero new inference in curation.
Protocol: 128 sampled frames, 64 thumbnails, spatial max_tiles_video=48,
NVILA-8B-HD-Video, bfloat16, greedy, exact fine-only K16, no selector EOS.
HLVid official benchmark test is distinct from the protected human-gaze test;
this curation does not access the latter. HLVid has prior historical exposure.

Primary micro QA: trained equal-seed mean 50.3109%;
pretrained 50.0000%; Center16 51.1194%. These small differences do not establish
superiority. Macro-video is a declared sensitivity, not a replacement primary.
`contrast_summary.csv` reports separate 90% training-seed Student-t intervals
(six independent endpoints, df5) and paired 77-video cluster bootstrap intervals
(10,000 iterations; conditional on those six seeds). Per-seed video intervals
are in `paired_per_seed.csv`. Neither uncertainty source replaces the other;
no combined interval or equivalence conclusion is asserted.

Human-gaze references ran on H100 NVL; controls on A40. Cross-hardware QA is
explicitly approved as descriptive, but timing is hardware-specific. Allocation
cost includes preempted segments: pretrained 18.4192 A40 hours;
Center16 15.6906 A40 hours. Logical completed QA
is not the number of event attempts. Missing historical counters remain missing;
current-process-only resumed summaries are not complete resource measurements.
`control_per_example_telemetry.csv` separates acquisition, selector, encoder,
generation/language timing, actual context and valid versus padded patch counts.
Process warmup is not a video-cache-hit label; these controls re-decode per QA.
Long expanded contexts are retained without truncation, not capped to nominal
40,960/32,768 metadata. A clean new decode audit supports, but cannot prove,
all historical reads. No reference QA rerun is warranted merely for telemetry.

Reproduce from repository root with the committed curation config and the
Linux raw paths in manifest.json, using the established healthy CPU interpreter:
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-project /home/stud/latn/miniconda3/envs/vila-autogaze-eval/bin/python scripts/human_gaze/curate_hlvid_k16_controls.py --config experiments/human_gaze/configs/r2e_hlvid_k16_causal_curation_v1.yaml --output-dir <fresh-directory>`.
The script verifies identity/score/hash/population, streams large sidecars,
and refuses existing outputs. Original control admission/preflight and shared
audit identities are embedded in the manifest. Heavy action sidecars remain
on Linux; no videos, frames, model weights or per-example answer caches in Git.
