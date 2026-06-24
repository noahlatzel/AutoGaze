# Experiment Metrics

This file defines the metrics used in temporal proxy, delta-policy, and
HLVid/NVILA experiments. Metric implementations should live in code; this file
records meaning, expected direction, and interpretation.

## Proxy Scores

| metric | implementation | direction | definition |
| --- | --- | --- | --- |
| `pixel_mse` | `autogaze.temporal_sampling.thumbnail_metrics` | larger means more visual change | mean squared RGB difference between aligned thumbnail frames |
| `edge_l1` | `autogaze.temporal_sampling.thumbnail_metrics` | larger means more edge change | mean absolute difference between Sobel edge magnitudes |
| `ssim_distance` | `autogaze.temporal_sampling.thumbnail_metrics` | larger means less structural similarity | `1 - global_ssim` on thumbnail frames |
| `mean_pairwise_change` | `autogaze.temporal_sampling.thumbnail_metrics` | larger means more aligned change | mean aligned pairwise distance from current chunk to candidate chunk |
| `mean_novelty` | `autogaze.temporal_sampling.thumbnail_metrics` | larger means candidate frames are less redundant | mean nearest-neighbor distance from candidate frames to current chunk |
| `within_diversity` | `autogaze.temporal_sampling.thumbnail_metrics` | larger means candidate chunk is internally diverse | mean upper-triangle pairwise distance inside candidate chunk |
| `siglip_cosine` | `autogaze.temporal_sampling.semantic_metrics` | larger means more semantic change | aligned mean cosine distance between SigLIP frame embeddings |
| `dinov2_cosine` | `autogaze.temporal_sampling.semantic_metrics` | larger means more semantic change | aligned mean cosine distance between DINOv2 frame embeddings |
| `siglip_qgain` | `autogaze.temporal_sampling.semantic_metrics` | larger means more question relevance gain | aligned mean increase in SigLIP image/text relevance |
| `semantic_mean_novelty` | `autogaze.temporal_sampling.semantic_metrics` | larger means semantic novelty | nearest-neighbor cosine novelty of candidate embeddings |
| `semantic_diversity` | `autogaze.temporal_sampling.semantic_metrics` | larger means semantic diversity | upper-triangle pairwise cosine distance inside candidate embeddings |
| `question_relevance` | `autogaze.temporal_sampling.semantic_metrics` | larger means candidate contains question-relevant evidence | max SigLIP image/text relevance in candidate chunk |

Action cost is not part of the metric definition. Selection utilities are:

```text
utility(delta) = proxy_score(delta) - lambda_cost * temporal_cost(delta)
```

Supported temporal costs are `none`, `linear`, and `log`.

## HLVid/NVILA Metrics

| metric | direction | meaning |
| --- | --- | --- |
| `accuracy` | higher is better | multiple-choice answer accuracy over HLVid examples |
| `by_category.*.accuracy` | higher is better | category-specific HLVid accuracy |
| `mean_visited_tiles` | lower is cheaper | average number of temporal tiles passed to NVILA |
| `mean_visited_frames` | lower is cheaper | average visited frames, usually `mean_visited_tiles * tile_len` |
| `mean_llm_inference_seconds` | lower is cheaper | mean NVILA generation time per example |
| `mean_proxy_seconds_per_score` | lower is cheaper | mean proxy scoring time for one candidate action |
| `selected_delta_counts` | descriptive | empirical action distribution |

## Delta-Policy Metrics

| metric | direction | meaning |
| --- | --- | --- |
| `accuracy` | higher is better | exact pseudo-label match |
| `top2_accuracy` | higher is better | pseudo-label in top two predicted classes |
| `macro_f1` | higher is better | class-balanced F1 over delta labels |
| `balanced_accuracy` | higher is better | recall averaged over classes |
| `mean_abs_delta_error` | lower is better | absolute frame-delta error |
| `mean_abs_seconds_error` | lower is better | delta error converted using effective FPS |
| `mean_reward` | higher is better | contextual-bandit reward under logged all-action scores |
| `mean_delta` | descriptive | average selected frame delta |
| `prediction_counts` | descriptive | selected/predicted delta distribution |

## Reporting Rule

Every experiment summary should report at least one quality metric, one compute
metric, and one policy-behavior metric. For example:

```text
quality: accuracy or reconstruction/continuity error
compute: visited frames, visited tiles, timing
behavior: delta distribution, frame-mask distribution, or label distribution
```
