# Training AutoGaze

AutoGaze is trained in a 2-stage pipeline:
1. **Stage 1: Pre-training with Next Token Prediction (NTP)** — AutoGaze learns to predict ground-truth gazing sequences via next-token prediction.
2. **Stage 2: Post-training with Reinforcement Learning (RL)** — AutoGaze is further trained with GRPO using reconstruction reward to discover better gazing sequences.

## Prerequisite: Download Data and Model Weights

### Training Data

Download and extract the training data from [bfshi/AutoGaze-Training-Data](https://huggingface.co/datasets/bfshi/AutoGaze-Training-Data):

```bash
# Using huggingface-cli (recommended)
hf download bfshi/AutoGaze-Training-Data --repo-type dataset --local-dir AutoGaze-Training-Data

# Extract the tar.gz archives
cd AutoGaze-Training-Data
for f in *.tar.gz; do tar -xzf "$f"; done
```

### VideoMAE Weights

Download the VideoMAE model used as the reconstruction task model from [bfshi/VideoMAE_AutoGaze](https://huggingface.co/bfshi/VideoMAE_AutoGaze):

```bash
hf download bfshi/VideoMAE_AutoGaze --local-dir VideoMAE_AutoGaze
```

### Expected File Structure

After downloading and unzipping, your data directory should look like this:

```
AutoGaze-Training-Data/
├── InternVid_res448_250K/
│   ├── train/
│   │   ├── xxx.mp4
│   │   ├── yyy.mp4
│   │   └── ...
│   └── val/
│       ├── zzz.mp4
│       └── ...
├── 100DoH_res448_250K/
│   ├── train/
│   │   └── ...
│   └── val/
│       └── ...
├── Ego4D_res448_250K/
│   ├── train/
│   │   └── ...
│   └── val/
│       └── ...
├── scanning_SAM_res448_50K/
│   ├── train/
│   │   └── ...
│   └── val/
│       └── ...
├── scanning_idl_res448_50K/
│   ├── train/
│   │   └── ...
│   └── val/
│       └── ...
└── gazing_labels.json        # Ground-truth gazing sequences for NTP pre-training

VideoMAE_AutoGaze/
├── videomae.pt               # Pre-trained VideoMAE weights (2 GB)
├── config.yaml
└── ...
```

Each video sub-dataset contains `train/` and `val/` splits, with `.mp4` video files inside. The `gazing_labels.json` file contains pre-computed ground-truth gazing sequences used in Stage 1 (NTP pre-training).

## Stage 1: Pre-training with NTP

Pre-train AutoGaze by learning to predict ground-truth gazing sequences via next-token prediction. See `scripts/examples/example_ntp_training.sh`:

```bash
torchrun \
    --nnodes=1 --nproc_per_node=8 \
    -m autogaze.train \
        --config-name video_folder_video_mae_reconstruction_ar_gaze_ntp \
        dataset.root=\'<path to InternVid_res448_250K>,<path to 100DoH_res448_250K>,<path to Ego4D_res448_250K>,<path to scanning_SAM_res448_50K>,<path to scanning_idl_res448_50K>\' \
        dataset.gt_gazing_pos_paths.train=\'<path to gazing_labels.json>\' \
        dataset.clip_len=16 \
        model.gazing_ratio_config.sample_strategy_during_training=fixed \
        model.gazing_ratio_config.sample_strategy_during_inference=fixed \
        model.gazing_ratio_config.fixed.gazing_ratio=0.1 \
        model.gazing_ratio_each_frame_config.sample_strategy_during_inference=dirichlet \
        model.gazing_ratio_each_frame_config.dirichlet.alpha=\'10,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3\' \
        model.scales=32+64+112+224 \
        model.num_vision_tokens_each_frame=265 \
        model.has_task_loss_requirement_during_training=False \
        model.has_task_loss_requirement_during_inference=False \
        model.gaze_model_config.gaze_decoder_config.num_multi_token_pred=10 \
        task.recon_model=facebook/vit-mae-large \
        task.recon_sample_rate=0.125 \
        task.recon_model_config.loss_type=l1+dinov2_reg+siglip2 \
        task.recon_model_config.loss_weights=1+0.3+0.3 \
        task.scales=32+64+112+224 \
        algorithm.optimize_task_loss_prediction=True \
        trainer.train_gaze=True \
        trainer.train_task=False \
        trainer.detach_task=True \
        trainer.lr=5e-4 \
        trainer.lr_schedule=linear \
        trainer.optimizer=adam \
        trainer.n_epochs=150 \
        trainer.batch_size=1024 \
        trainer.per_gpu_max_batch_size=32 \
        trainer.temp_schedule_args.exp.temp_end=0.3 \
        trainer.temp_schedule_args.exp.temp_start=3 \
        trainer.val_nsteps=3000 \
        trainer.save_nsteps=500 \
        trainer.task_weights=<path to VideoMAE_AutoGaze>/videomae.pt \
        trainer.exp_name=example_ntp_training
```

Replace all `<path to ...>` placeholders with your actual paths.

## Stage 2: Post-training with RL

After NTP pre-training, further train AutoGaze with GRPO (Group Relative Policy Optimization) using reconstruction reward. See `scripts/examples/example_rl_training.sh`:

```bash
torchrun \
    --nnodes=1 --nproc_per_node=1 \
    -m autogaze.train \
        --config-name video_folder_video_mae_reconstruction_ar_gaze_grpo \
        dataset.root=\'<path to InternVid_res448_250K>,<path to 100DoH_res448_250K>,<path to Ego4D_res448_250K>,<path to scanning_SAM_res448_50K>,<path to scanning_idl_res448_50K>\' \
        dataset.clip_len=16 \
        model.gazing_ratio_config.sample_strategy_during_training=fixed \
        model.gazing_ratio_config.sample_strategy_during_inference=fixed \
        model.gazing_ratio_config.fixed.gazing_ratio=0.75 \
        model.gazing_ratio_each_frame_config.sample_strategy_during_training=self \
        model.scales=32+64+112+224 \
        model.num_vision_tokens_each_frame=265 \
        model.has_task_loss_requirement_during_training=False \
        model.has_task_loss_requirement_during_inference=True \
        model.task_loss_requirement_config.sample_strategy_during_training=uniform \
        model.task_loss_requirement_config.sample_strategy_during_inference=fixed \
        model.task_loss_requirement_config.fixed.task_loss_requirement=0.7 \
        model.task_loss_requirement_config.uniform.task_loss_requirement_min=0.5 \
        model.task_loss_requirement_config.uniform.task_loss_requirement_max=1.0 \
        model.gaze_model_config.gaze_decoder_config.num_multi_token_pred=10 \
        task.recon_model=facebook/vit-mae-large \
        task.recon_sample_rate=0.125 \
        task.recon_model_config.loss_type=l1+dinov2_reg+siglip2 \
        task.recon_model_config.loss_weights=1+0.3+0.3 \
        task.scales=32+64+112+224 \
        algorithm.group_size=12 \
        algorithm.discount_factor=0.995 \
        algorithm.optimize_task_loss_prediction=True \
        trainer.train_gaze=True \
        trainer.train_task=False \
        trainer.detach_task=True \
        trainer.lr=5e-4 \
        trainer.n_epochs=1 \
        trainer.batch_size=64 \
        trainer.per_gpu_max_batch_size=2 \
        trainer.temp_schedule_args.exp.temp_start=3 \
        trainer.temp_schedule_args.exp.temp_end=0.3 \
        trainer.val_nsteps=1000 \
        trainer.save_nsteps=500 \
        trainer.task_weights=<path to VideoMAE_AutoGaze>/videomae.pt \
        trainer.gaze_weights=<path to NTP pre-training exp>/checkpoint_latest_gaze \
        trainer.exp_name=example_rl_training
```

## Parameter Explanation

### Dataset Parameters

| Parameter | Description |
|-----------|-------------|
| `dataset.root` | Comma-separated paths to video datasets. Each path should contain `train/` and `val/` subdirectories with `.mp4` files. |
| `dataset.gt_gazing_pos_paths.train` | Path to the ground-truth gazing label JSON file. Only needed for NTP pre-training. |
| `dataset.clip_len` | Number of frames to sample from each video clip. |

### Model Parameters

| Parameter | Description |
|-----------|-------------|
| `model.scales` | Multi-scale patch sizes separated by `+` (e.g., `32+64+112+224`). AutoGaze selects patches from each of these spatial scales, allowing it to use coarser patches for simple regions and finer patches for detailed regions. |
| `model.num_vision_tokens_each_frame` | Total number of vision tokens (across all scales) per frame. This is the maximum number of patches that can be selected for each frame. |
| `model.gaze_model_config.gaze_decoder_config.num_multi_token_pred` | Number of tokens predicted in parallel by the autoregressive decoder. Higher values (e.g., 10) can speed up inference but performance will drop if it's too high. |

**Gazing ratio** controls how many patches to select in the **whole video**:

| Parameter | Description |
|-----------|-------------|
| `model.gazing_ratio_config.sample_strategy_during_training` | How gazing ratio is sampled during training (`fixed`, `uniform`, or `exponential`). |
| `model.gazing_ratio_config.sample_strategy_during_inference` | How gazing ratio is sampled during inference. |
| `model.gazing_ratio_config.fixed.gazing_ratio` | The fixed gazing ratio to use. E.g., `0.1` means selecting 10% of patches. |

**Gazing ratio distribution across frames** controls how the total gazing budget is distributed among frames:

| Parameter | Description |
|-----------|-------------|
| `model.gazing_ratio_each_frame_config.sample_strategy_during_training` | Strategy for distributing gazing budget across frames during training. `uniform` distributes uniformly; `dirichlet` uses a Dirichlet distribution; `self` will first run the model with a task loss constraint and then record the number of patches it gazes at for each frame, and then use that as the gazing ratios for each frame (such that the gazing ratios are on-policy). |
| `model.gazing_ratio_each_frame_config.dirichlet.alpha` | Dirichlet concentration parameters, one per frame. Higher values mean more budget for that frame. E.g., `'10,3,3,...,3'` allocates more budget to the first frame. |

**Task loss requirement** enables early stopping when reconstruction quality is sufficient (primarily used during RL):

| Parameter | Description |
|-----------|-------------|
| `model.has_task_loss_requirement_during_training` | Whether to condition the model on a task loss requirement during training. |
| `model.has_task_loss_requirement_during_inference` | Whether to enable early stopping based on reconstruction quality during inference. |
| `model.task_loss_requirement_config.fixed.task_loss_requirement` | The fixed task loss threshold (0-1). Lower values require better reconstruction quality and thus more patches. `0.7` is a good default. |
| `model.task_loss_requirement_config.uniform.*` | Min/max range for uniformly sampling the task loss requirement during training, which helps the model generalize to different quality requirements. |

### Task Parameters

| Parameter | Description |
|-----------|-------------|
| `task.recon_model` | The VideoMAE model used for reconstruction (e.g., `facebook/vit-mae-large`). |
| `task.recon_sample_rate` | Fraction of frames to use for computing reconstruction loss. Lower values (e.g., `0.125`) reduce computation. |
| `task.recon_model_config.loss_type` | Reconstruction loss type(s) separated by `+`. `l1` is pixel-level L1 loss; `dinov2_reg` is DINOv2 feature-level loss; `siglip2` is SigLIP2 feature-level loss. Combining multiple losses improves reconstruction quality. |
| `task.recon_model_config.loss_weights` | Corresponding weights for each loss type, separated by `+` (e.g., `1+0.3+0.3`). |
| `task.scales` | Specifies which scales the task model operates on. Note that it **doesn't** need to match `model.scales` because AutoGaze can process videos with any spatial dimensions by splitting a video into fixed-size tiles.  |

### Algorithm Parameters

| Parameter | Description |
|-----------|-------------|
| `algorithm.optimize_task_loss_prediction` | Whether to jointly train the model to predict the task loss at each gazing step. This enables the model to estimate when reconstruction quality is sufficient for early stopping. |
| `algorithm.group_size` | **(RL only)** Number of sampled gazing sequences per input in GRPO. Each input is copied `group_size` times, and the model samples different gazing sequences. Advantages are computed relative to the group mean. Larger values give more stable training but increase memory. |
| `algorithm.discount_factor` | **(RL only)** Temporal discount factor for the GRPO advantage. Values close to 1.0 (e.g., `0.995`) mean rewards are attributed more evenly across the gazing trajectory; lower values emphasize more recent gazing decisions. |

### Trainer Parameters

| Parameter | Description |
|-----------|-------------|
| `trainer.train_gaze` | Whether to train the gaze model. Should be `True` for both stages. |
| `trainer.train_task` | Whether to train the task model (VideoMAE). Set to `False` since we use a frozen pre-trained VideoMAE. |
| `trainer.detach_task` | Whether to run the task model under `torch.no_grad()`. Set to `True` to save memory when the task model is frozen. |
| `trainer.task_weights` | Path to pre-trained VideoMAE weights (`videomae.pt`). |
| `trainer.gaze_weights` | Path to pre-trained gaze model checkpoint. Used in Stage 2 to load the NTP pre-trained model. |
| `trainer.lr` | Learning rate. |
| `trainer.lr_schedule` | Learning rate schedule (`linear`, `linear_w_warmup`, or `constant`). |
| `trainer.optimizer` | Optimizer (`adam` or `sgd`). |
| `trainer.n_epochs` | Number of training epochs. Stage 1 uses many epochs (150) for NTP convergence; Stage 2 uses a few epochs due to the large computational cost of RL. |
| `trainer.batch_size` | Global batch size across all GPUs. |
| `trainer.per_gpu_max_batch_size` | Maximum batch size per GPU. Gradient accumulation is automatically applied if `batch_size > per_gpu_max_batch_size * num_gpus`. |
| `trainer.temp_schedule_args.exp.temp_start` / `temp_end` | Temperature annealing schedule for the autoregressive decoder's sampling distribution. Starts high (more exploration) and decays to a lower value (more exploitation). |
| `trainer.val_nsteps` | Run validation every this many training steps. |
| `trainer.save_nsteps` | Save a checkpoint every this many training steps. |
| `trainer.exp_name` | Experiment name. Checkpoints are saved under `exps/<exp_name>/`. |

