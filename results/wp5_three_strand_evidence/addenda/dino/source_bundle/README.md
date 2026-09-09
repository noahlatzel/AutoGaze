# DINOv3 frozen-panel evidence

Completed raw observations only; no new inference or measurement. Twelve frozen validation videos, two per each of six sources, with 16 frames/video. Fidelity means weight the six sources equally.

| Arm | Final patches | Encoder GFLOPs/frame | Matching GFLOPs/frame | CLS distance | Batch-16 amortized ms/frame | Batch-one ms/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 196 | 9.396210 | 0.000000 | 0.000000 | 3.2273 | 11.0224 |
| Human K16 mean (6 endpoints) | 16 | 1.015548 | 0.000000 | 0.469377 | 8.1152 | 8.5589 |
| Random K16 mean (16 masks) | 16 | 1.015548 | 0.000000 | 0.480197 | 8.1495 | 8.6528 |
| Center K16 | 16 | 1.015548 | 0.000000 | 0.481204 | 8.1083 | 8.6296 |
| Pretrained K16 | 16 | 1.015548 | 0.000000 | 0.491959 | 8.1002 | 8.5545 |
| ToMe r=5 | 136 | 7.902465 | 0.011017 | 0.020375 | 2.9978 | 22.0718 |
| ToMe r=10 | 76 | 6.447581 | 0.008092 | 0.070553 | 2.5139 | 21.9480 |
| ToMe r=15 | 16 | 5.031558 | 0.005976 | 0.193504 | 2.0030 | 21.8804 |
| Temporary spatial r=15 | 196 | 8.652202 | 0.003355 | 0.008902 | unavailable | 33.1535 |

Encoder and matching counts are analytical matrix-multiply FLOPs (FMA=2), not a complete operator profiler. The batch-one random entry uses only the single declared mask seed, not the 16-replicate mean. Human timing entries average six endpoint medians. The two timing columns have separate batching protocols and panels; batch-one means use only two predeclared videos. Mask selection is excluded from encoder timings.

## Paired K16 comparisons

| Human mean minus control | CLS-distance difference | Descriptive 90% interval |
| --- | ---: | ---: |
| Pretrained K16 | -0.022582 | [-0.055264, +0.009073] |
| Random K16 mean (16 masks) | -0.010820 | [-0.035836, +0.012935] |
| Center K16 | -0.011827 | [-0.021247, -0.002797] |

Post-hoc descriptive 90% paired source-stratified video bootstrap interval; 10,000 draws, seed 20260909; conditional on the six fixed sources, six observed training endpoints and 16 observed random-mask replicates. Frames, seeds and timing repeats are not resampled.

The three comparisons are exploratory and unadjusted. Lower CLS distance means closer encoder representation, not better downstream accuracy. Observed mean advantages over pretrained and random controls have video intervals spanning zero; the center contrast is below zero under this conditional descriptive analysis.

## Measurement boundaries

ToMe is faster than dense in the measured batch-16 protocol and slower in the measured batch-one protocol. The K16 path processes per-image ragged sequences internally. These observations concern this implementation and the recorded RTX A2000 environment. No mechanism-specific explanation or end-to-end speedup is established.

Temporary spatial merging restores 196 patches after each block; cumulative ToMe r=15 finishes at 16 patches. They are different interventions. Their full-panel fidelity and analytical costs can be compared, while spatial latency is available only in the batch-one supplement.

Existing single-pass exporter timing is preserved in exporter_observations.csv and exporter_observation_summary.csv. Those observations have their own environment, no warmup/repetition, and no timing interval. They are never added to controlled encoder timing. Allocator tables measure PyTorch live-byte peaks, not process-wide GPU memory.

The manifest seals the raw-input hashes, source identity, all tables, figure sources, captions and nine image files. All raw source summaries and 14 packaged metric arms were reproduced. No RGB arrays or weights were available locally for fresh hashing.
