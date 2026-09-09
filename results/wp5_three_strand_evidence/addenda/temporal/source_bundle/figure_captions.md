# Temporal figures

Re-rendered from the immutable compact tables; no new inference or bootstrap.

## temporal_accuracy_frames_frontier

Selected historical HLVid temporal-selection settings at maximum spatial tile budget 48. Panel a plots question-mean retained candidate-grid frames against video-macro QA accuracy (77 equally weighted videos); only point estimates are shown. Gray points connect the fixed-gap sweep on the 128-point grid. Panel b shows paired differences against each setting's same-grid uniform reference, with 90% paired video-bootstrap intervals from 10,000 replicates. These are intervals for differences, not marginal accuracy intervals. The Pixel-MSE .04 setting is an exposed exploratory threshold; the full sweep appears separately. Repeated HLVid test inspection and unmatched frame budgets preclude confirmed threshold or equal-budget superiority claims. The learned policy uses maximum spatial tile budget 1 and is excluded from this figure. Candidate-grid frames are not raw decoded source frames.

## pixel_threshold_sweep

The complete recorded Pixel-MSE threshold sweep on a 256-point candidate grid. Panel a shows question-mean retained frames. Panel b shows video-macro QA accuracy with marginal 90% video-bootstrap intervals. Panel c shows paired video-macro accuracy differences against Uniform 256 with 90% paired video-bootstrap intervals; all intervals use 10,000 replicates. Dashed lines mark the uniform reference. QA is non-monotonic as the retained-frame count decreases. Because these thresholds were repeatedly inspected on the HLVid test, the apparent peak at .04 is exploratory.

## historical_h100_timing

Historical per-question timing for all recorded temporal proxy/uniform conditions labeled NVIDIA H100 NVL, separated into cold-decode and frame-cache-hit strata. Blue proxy time is included within each gray-plus-blue total; it is not added a second time. Labels show mean total seconds and the actual number of questions in that stratum. Counts range from 98 to 100 cold-decode questions and 168 to 170 cache-hit questions, summing to 268 per condition. The strata have different video/question composition, and query-free proxy scores were recomputed per question. These historical records do not establish a controlled cache effect or end-to-end speedup. DINOv2 diversity is excluded because it used an RTX PRO 6000 Blackwell GPU.

## learned_delta_schedule

The learned DINO-delta policy realized one schedule across all 77 HLVid videos (268 questions): 16-frame blocks starting at candidate-grid positions 0, 48 and 96, with recorded actions 32, 32 and 16. Blue blocks show the 48 retained positions; gray spans show the following gaps. Delta measures a gap after the kept block, so the next start is current start + 16 + delta. This mtv1 deployment establishes fixed-schedule collapse, not adaptive acquisition. Its spatial resolution budget differs from the mtv48 uniform/fixed-gap QA references, and no matched uniform-48 QA comparator is available. Candidate-grid indices are not raw source-frame indices.
