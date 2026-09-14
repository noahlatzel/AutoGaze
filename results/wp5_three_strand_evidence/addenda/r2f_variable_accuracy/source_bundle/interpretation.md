# Complete R2f paired accuracy; variable costs still missing

All six QA endpoints are complete and validated (268 ordered unique questions
per arm; 77 videos; three independent paired training seeds). They reuse the
existing official HLVid test split/protocol; this is not a newly unseen benchmark.
R2d endpoints were fixed at 20,000 training updates, and EOS biases were selected
on 48 source-balanced training clips, not HLVid answers. No new policy training,
checkpoint selection, answer generation, calibration or protocol variation was
performed for this curation. Human-gaze protected test remains unopened here.

| Estimand | Variable EOS | Forced K16 | Paired difference | Video-cluster 90% interval | Seed t95 interval |
|---|---:|---:|---:|---:|---:|
| Video-macro (primary) | 48.295% | 50.042% | −1.746 pp | [−5.563, +1.754] pp | [−5.319, +1.827] pp |
| Question-micro (secondary) | 48.383% | 50.124% | −1.741 pp | [−4.852, +1.379] pp | [−2.276, −1.206] pp |

Variable-minus-forced primary differences are −1.619, −3.244 and −0.376 pp for
base seeds 440826–828. Correct counts are respectively 132/137, 130/135 and
127/131 (variable/forced, each /268). The smaller seed spread for question-micro
does not replace the primary metric or eliminate video-level uncertainty.
The paired video intervals include zero: report a bounded negative descriptive
deployment contrast, not a definitive population effect or a positive story.

This comparison is **not allocation-only**: actual EOS stopping changes later
spatial ordering too. R2d's earlier allocation-only validation analysis holds
the common forced-K36 spatial ordering fixed while comparing actual versus
same-source shuffled lengths. That small lift (+0.007764 coverage) does not
overcome R2d's in-domain degradation (variable 0.378752 vs forced 0.417117) or
calibration target miss (validation K15.156 vs accepted 15.5–16.5).

Actual HLVid variable action, recovered-patch and expanded-token/context costs
are **null** pending validated replay. Training calibration lengths and resumed
process-local tail means are not full benchmark costs. Forced K16's nominal
fine-action length is 16 and recovered-patch length is 64 under the adapted
tile/recovery exact-K contract; expanded context counts are unavailable there
too. No measured efficiency/speedup is claimed.

## Compute and provenance

`scheduler/` preserves the sole owner's same-A40 node-pin override receipt
(source d5c7ae6) and every `sacct -D` allocation/batch segment. QA consumed
361,919 reserved A40 seconds (100.533 GPU-lane hours), including preempted
segments; batch/extern steps are not double-counted. These are reservation
walltimes, not measured CUDA utilization. Six final QA segments completed on
node17 except variable440827, which finished its resumed tail on node15.

Original processor replay 1700681 failed on node16 after 17 allocated seconds,
before any observation or answer generation (batch MaxRSS 532376 K). Its frozen
ecd72f source, log hash and empty artifacts are preserved. Its one-dimensional
path serialization repair is a new immutable d3b8d5a snapshot, not a rewrite.

Completed forced440827 is an uninterrupted full-protocol A40 execution, with
17:41:39 reservation walltime and batch MaxRSS 131296872 K (host memory, not
VRAM). Its adapter records A40 capacity 47,696,969,728 bytes (~44.42 GiB),
**not a GPU-memory peak**. No historical peak-VRAM counter is available in this
bundle. Full-protocol completion supports feasibility but is not a precise
VRAM profile or proof for another policy's variable sequence lengths.

`manifest.json` hashes source QA/summaries, checkpoint/calibration identities,
compact tables/figures, original R2d metrics, CPU construction smoke and scheduler
receipt. The CPU smoke has no CUDA allocation or EOS-semantics certification.
Use `paired_aggregate.csv` and `paired_videos.csv` as figure/table sources;
`paired_questions.csv` preserves all 804 paired QA contrasts.

The first figure draft is archived unchanged at
`/storage/user/latn/artifacts/autogaze-r2f/20260914-0155_r2f-hlvid-complete-paired-accuracy_13054f8`.
This publication changes figure layout/provenance packaging, not QA or metrics.
