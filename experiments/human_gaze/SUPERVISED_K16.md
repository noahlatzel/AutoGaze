# Supervised human gaze versus coverage-reward RL

This is the user-authorized six-seed K16 comparison added on 10 September 2026.
The initial implementation is ready for real-model preflight; no trained result
is claimed. The executable study contract is `configs/supervised_k16_comparison.yaml`.
The remote master-thesis task **🔄 Supervised human-gaze comparison** owns the
protocol and evidence. The fixed-budget task owns all Slurm admission.

## Scientific question and objective

Current K16 RL uses human heatmap coverage as a scalar reward, four sampled
trajectories per clip and a zero KL coefficient. Its optional KL regularizer
compares the current policy with a frozen reference policy, not human gaze.

The new baseline fits the human mass over remaining fine cells with soft-target
cross-entropy, equivalent in gradient to forward KL from human mass to policy.
At every selection, remove the **previous** selections from both supports and
renormalize. Sample teacher histories from human mass without replacement and
score those supplied histories with the existing autoregressive model. These
are annotation-derived spatial sequences, not observed human scanpaths.
Training uses one teacher trajectory per clip and no policy rollout groups.
Evaluation remains greedy exact K16. This is a comparison of training methods
including their history distributions, not an isolated gradient-estimator study.

The stable supervised probability output uses the same ten prediction-head
alignment as inference, float32 normalization and exact zero support for prior
selections. It is opt-in; existing RL and evaluation paths retain their behavior.
Exhausted human mass produces uniformly sampled legal continuation actions to
finish K16, but those rows have zero distribution loss and are counted explicitly.

## Matching and interpretation

Use the same six base seeds, shared original pretrained model, data and
preprocessing, source/video-balanced sampler, frozen visual encoder/connector,
trainable decoder, batch of four base clips and 20,000-update exposure ceiling.
The first 2,315 updates reproduce the executable historical LR schedule; the
second phase starts fresh Adam and performs 17,685 updates at 3e-6. No Adam reset
occurs at cumulative 10k. The second-phase config requires an explicit verified
checkpoint from the **same seed's supervised first phase**, supplied as
`trainer.gaze_weights` or `SUPERVISED_STAGE1_CHECKPOINT`. It must never load an RL
post-trained endpoint as its initialization.

Historical stage-2/3 RL resumed at the next epoch boundary; common seeds and a
common sampler do not establish identical historical clip order. Match the
declared data distribution and exposure counts, and retain this limitation.
Record actual resolved configs, initialization/checkpoint/data hashes and the
execution environment before training. A preflight must check exact fine K16,
finite forward/backward gradients, decoder-only parameter updates, teacher-RNG
and model resume, actual GPU/host memory, and throughput at the requested batch.

The main comparison uses all six fixed 20k endpoints. Earlier prescribed 5k,
10k and 15k checkpoints describe convergence. Report base clips, trajectories,
action evaluations, validation cost and measured hardware-specific time; the
planned 75% reduction in trajectories is not a measured fourfold speedup.

Use source-macro, video-aggregated validation coverage as primary, with all seed
results and separate training-seed and paired-video uncertainty. Center bias
alone is not collapse: compare dynamic selections against static-frequency
Top16 and same-source shuffled-video controls, as well as Center16 and the
train-derived source prior. Report center overlap, entropy and SL–RL selection
agreement, contextualized by RL–RL between-seed agreement. Reuse the existing
24 human-selected off-center examples and their predetermined base seed 440826;
aggregate off-center results over all six seeds. Existing K24/K32 results are
budget context, not new supervised training arms.

Validation is historically explored. The test split was also evaluated in
R0/R1, so a later frozen descriptive test pass must acknowledge that history.
Neither test nor HLVid may select training configurations or favorable seeds.

## Conditional transfer evaluation

The YAML fixes a practical HLVid gate: comparable coverage with less scheduled
training work and complete resource accounting, or useful overall/off-center
improvement, together with positive dynamic-versus-static/shuffled evidence.
The dynamic-control condition uses the equal-seed mean of source/video-macro
coverage, not a requirement for every individual seed. Route A's nominal action
row reduction is structural; actual elapsed time can differ and must be reported.
Passing that route does not establish faster training.
This is a decision to measure transfer, not a formal non-inferiority or
significance claim. If the gate passes, evaluate all six fixed endpoints under
the established full HLVid recipe, including MTV48, 128 frames,64 thumbnails and
exact K16. Cross-hardware QA is user-approved; latency remains hardware-specific.

No new job is admitted by the YAML. Memory, throughput, artifact locations,
immutable execution SHA and scheduler limits must be supplied to the existing
sole admission owner. Existing jobs 1690935, 1701485 and 1700681 retain priority.

## Focused validation

Run the new human-gaze NTP, supervised model distribution and trainer-contract
tests together with the existing fine-only generation and GRPO KL tests. They
cover target support and gradients, prediction-head/frame alignment, numerical
stability, config matching and checkpoint restoration. CPU checks are not a
substitute for a real-model GPU forward/backward and memory preflight.

Initial integrated validation: **77 tests passed** on Python3.12/PyTorch2.9 CPU.
Supervised checkpoints preserve the actual next sampler position and dedicated
teacher RNG. A completion marker checks model/optimizer/teacher-state hashes
and the phase/sampler contract before loading; an interrupted or mixed latest
bundle is rejected. Periodic snapshots are not automatic recovery targets.
The immutable execution manifest must additionally bind source and dataset hashes.

## Production execution

The bounded real-model receipt under
`results/supervised_k16_real_preflight/20260909-230035_supervised-k16-real-preflight_21e9ab8`
passed six single-rank NCCL forward/backward iterations and six label-free greedy
K16 generations. It measured 407,873,536 bytes peak CUDA allocation and
2,149,191,680 bytes process RSS on the RTX A2000 VM. An extra GPU smoke for an
Adam step or batch-16 validation was not required: Adam state for 2.04M FP32
trainable parameters is small, checkpoint restoration has CPU invariant tests,
and the identical validation batch size already completed in the historical
RTX5000 training runs.

`configs/supervised_k16_execution.yaml` is the production resource and phase
contract. `slurm/run_supervised_k16_comparison_array.sbatch` runs one paired seed
per array element, serializes stage one, its completion verification, stage two,
and its completion verification, and cannot run HLVid. The initial request is a
conservative `%1` A40 lane with one GPU, five CPUs, 32 GiB host RAM and a 24-hour
ceiling per element. The launcher requires a clean detached admitted commit,
verifies all input hashes before training, refuses existing output directories,
records phase timing and GPU telemetry, and emits an immutable per-seed execution
manifest. Final Slurm elapsed/GPU/MaxRSS accounting remains mandatory after each
allocation exits.
