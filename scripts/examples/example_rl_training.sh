export HYDRA_FULL_ERROR=1

torchrun \
    --nnodes=1 --nproc_per_node=1 \
    -m autogaze.train \
        --config-name video_folder_video_mae_reconstruction_ar_gaze_grpo \
        dataset.root=\'[path to InternVid_res448_250K data],[path to 100DoH_res448_250K data],[path to Ego4D_res448_250K data],[path to scanning_SAM_res448_50K data],[path to scanning_idl_res448_50K data]\' \
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
        trainer.task_weights=[path to VideoMAE_AutoGaze]/videomae.pt \
        trainer.gaze_weights=[path to pre-training exp]/checkpoint_latest_gaze \
        trainer.exp_name=example_rl_training