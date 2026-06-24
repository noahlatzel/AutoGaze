torchrun \
    --nnodes=1 --nproc_per_node=8 \
    -m autogaze.train \
        --config-name video_folder_video_mae_reconstruction_ar_gaze_ntp \
        dataset.root=\'[path to InternVid_res448_250K data],[path to 100DoH_res448_250K data],[path to Ego4D_res448_250K data],[path to scanning_SAM_res448_50K data],[path to scanning_idl_res448_50K data]\' \
        dataset.gt_gazing_pos_paths.train=\'[path to gazing_labels.json]\' \
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
        trainer.task_weights=[path to VideoMAE_AutoGaze]/videomae.pt \
        trainer.exp_name=example_ntp_training