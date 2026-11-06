# S1 supervised human-gaze K16 comparison

Status: complete 5-seed paired fixed-endpoint validation curation.

Primary source/video-macro K16 coverage: supervised 0.413900, RL 0.452249, paired delta -0.038350.
Human-only off-center paired coverage delta: -0.067557.

Practical HLVid gate: `closed_quality_only_amendment_no_qa_authorization`.

Intervals in `metrics.json` are descriptive and unadjusted; validation was historically explored.
Seed 440830 is a post-hoc operational attrition after scheduler preemption; excluded attempts and nominal exposure remain recorded, but complete resource totals are unavailable.
Nominal training trajectory action rows are an exposure/accounting quantity, not FLOPs or measured runtime.

November 5 quality-only amendment: missing scheduler receipts are retained in readiness.json; complete training resource totals are withheld. HLVid admission remains closed. Coverage and exposure summaries require all 30 verified exports. No training was resumed.

## Promotion note (2026-11-06)

Promoted from the S1 artifact directory after export array 1990042 (tasks 0–4) and curation 1990044 completed. Independent post-run validation (`validate_supervised.py` → `supervised-validation.json`) re-verified all 30 checkpoint/action bindings and reproduced both paired video-bootstrap intervals exactly.

| Checkpoint (cumulative updates) | Supervised coverage (5-seed mean) |
|---|---:|
| 2,315 | 0.4227 |
| 5,000 | 0.4255 |
| 10,000 | 0.4164 |
| 15,000 | 0.4116 |
| 20,000 | 0.4139 |
| RL 20k (reference) | 0.4522 |

Supervised − RL20k at matched base-clip exposure (80,000): −3.83 pp, paired video-bootstrap 90% [−5.03, −2.69]; off-center 110 clips −6.76 pp [−8.20, −5.30]. Seed 90% intervals also exclude zero. Supervised center overlap rises from 0.60 to 0.74 with training while off-center coverage falls.

Metadata correction: the inherited `convergence.warning` in `metrics.json` (kept byte-identical) says final scheduler totals include all attempts. That is wrong: the final scheduler receipts are missing and training resource totals are unavailable. See `promotion.json` for the original text, the corrected wording and hashes. No numbers changed.
The same inherited wording appears in the right panel of `figures/supervised_k16_convergence.*` ("See final scheduler totals"); no complete supervised scheduler totals exist. The figure is preserved unchanged.

Exposure reading (left panel): supervised is ahead at low exposure (0.423 at 9,260 base-clip presentations versus RL ≈0.39), RL overtakes near 28,000 presentations and keeps rising while supervised drifts toward the center. RL values come from logged validation during training; supervised values from the 25 new exports. Descriptive only.
