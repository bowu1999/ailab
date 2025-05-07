work_dir = "/path/ailab-workspace-250428"
experiment_name = "experiment_1"  # 你可以根据需要修改


cfg = dict(
    seed = 42,
    work_dir = work_dir,
    dist = dict(backend = "nccl", init_method = "env://"),
    workflow = [{"phase":"train"}, {"phase":"val"}],
    total_epochs = 300,
    hooks = dict(
        # MetricHook 会遍历子项并实例化各指标
        metrics = dict(
            type = "MetricHook",
            # 映射到 update(preds, targets)
            peak_signal_noise_ratio = dict(
                type = "MyPeakSignalNoiseRatio",
                data_range = 1.0,
                mapping = dict(
                    preds = "pred",    # model 输出键名
                    targets = "target"   # dataset 返回键名
                ),
                phases = ["val"]
            ),
        ),
        checkpoint = dict(type="CheckpointHook", interval = 1),
        resume = dict(
            type = "ResumeHook", 
            resume_from = ""
        ),
        lr_scheduler = dict(
            type = "LrSchedulerHook",
            scheduler = dict(
                type = "CosineAnnealingLR", 
                T_max = 10,
                eta_min = 0
            )
        ),
        logger = dict(
            type = "LoggerHook",
            interval = 200,
            log_dir = f"{work_dir}/logs",
            log_items = dict(
                lr = "optimizer.param_groups.0.lr",
                psnr = "meters.get('peak_signal_noise_ratio', None) and meters['peak_signal_noise_ratio'].avg"
            )
        ),
        ddp = dict(type = "DDPHook"),
        amp = dict(type = "AMPHook"),
        tensorboard = dict(type = "TensorboardHook"),
        wandb = dict(
            type = "WandbHook",
            init_args = dict(
                project = "my_project",
                name = experiment_name, 
                mode = "disabled"
            )
        )
    ),
    data = dict(
        train = dict(
            type = "MAEAnnFileDataset",
            annotation_file = "mae_train.txt"
        ),
        val = dict(
            type = "MAEAnnFileDataset",
            annotation_file = "mae_val.txt"
        ),
        train_dataloader = dict(
            batch_size = 16,
            num_workers = 16,
            pin_memory = True,
            persistent_workers = True,
            prefetch_factor = 4
        ),
        val_dataloader = dict(
            batch_size = 16,
            num_workers = 32,
            pin_memory = True,
            persistent_workers = True,
            prefetch_factor = 4
        ),
    ),
    model = dict(
        type = "MaskedAutoencoder",
        mapping = dict(  # call_fn 会把 data['image']、data['mask'] 传给 forward
            image = "image",
            mask = "mask"
        ),
        output_keys = ["pred"]  # 将模型唯一输出命名为 'pred'
    ),
    loss = dict(
        type    = "ReconstructionLoss",
        mapping = dict(
            pred = "pred",   # ReconstructionLoss.forward(pred=…,mask=…,target=…)
            mask = "mask",
            target = "target"
        )
    ),
    optimizer = dict(
        type = "AdamW",
        lr   = 1e-4,
        betas= (0.9, 0.999),
        eps  = 1e-6,
        weight_decay = 0.01
    )
)