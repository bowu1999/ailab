work_dir = "/path/ailab-workspace-250428"
experiment_name = "experiment_1"  # 你可以根据需要修改


cfg = dict(
    # 随机种子
    seed = 42,
    # 工作目录
    work_dir = work_dir,
    # 从哪个 checkpoint 恢复训练
    resume_from = "/path/ailab-workspace-250428/epoch_200.pth",
    # 分布式配置
    dist = dict(
        backend = "nccl",
        init_method = "env://",
    ),
    # 混合精度配置
    amp = dict(
        enabled = True,
        opt_level = "O1",
    ),
    # 流程配置：train / val / test
    workflow = [
        {"phase": "train"},
        {"phase": "val"},
        # {"phase": "test"}, 
    ],
    total_epochs = 100,
    # 钩子 (Hooks)
    hooks = dict(
        # 指标钩子：自动统计 top1/top5
        metrics = dict(
            type = "MetricHook",
            classification_accuracy=dict(type="ClassificationAccuracy"),
            regression_mse=dict(type="RegressionMSE"),
        ),
        # 检查点钩子：每隔多少 epoch 保存一次
        checkpoint = dict(
            type     = "CheckpointHook",
            interval = 1,
        ),
        # 断点续训
        resume = dict(
            type   = "ResumeHook",
            enable = True,
        ),
        # 学习率调度
        lr_scheduler = dict(
            type      = "LrSchedulerHook",
            scheduler = dict(type="CosineAnnealingLR", T_max=10,eta_min=0),
        ),
        # 日志钩子：控制台 & 文件
        logger = dict(
            type      = "LoggerHook",
            interval  = 200,
            log_dir   = f"{work_dir}/logs",
            log_items = dict(
                lr   = "optimizer.param_groups.0.lr",
                classification_accuracy = "meters['classification_accuracy'].avg",
                regression_mse = "meters['regression_mse'].avg",
            ),
        ),
        # 分布式并行
        ddp = dict(type="DDPHook"),
        # AMP 混合精度
        amp = dict(type="AMPHook"),
        # TensorBoard
        tensorboard = dict(type="TensorboardHook"),
        # W&B
        wandb = dict(
            type = "WandbHook",
            init_args = dict(
                project = "my_project",
                name = experiment_name,
                mode = "disabled",
            ),
        ),
    ),
    # 数据集与 Dataloader
    data = dict(
        train = dict(
            type = "ClsRegImageDataset",
            annotation_file = "/path/train.jsonl",
            x_key = "image_path",
            cls_key = "category_id",
            reg_key = "tr"
        ),
        val = dict(
            type = "ClsRegImageDataset",
            annotation_file = "/path/val.jsonl",
            x_key = "image_path",
            cls_key = "category_id",
            reg_key = "tr",
            dataset_type = "val"
        ),
        train_dataloader = dict(
            batch_size=16,
            num_workers=16, 
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        ),
        val_dataloader = dict(
            batch_size=16,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        ),
    ),
    # 模型
    model = dict(
        type = "MultiTaskResNet50",
        pretrained_weights_path = '/path/resnet50_a1_0-14fe96d1.pth',
        num_classes = 12, 
        regression_output_size = 1
    ),
    # 损失
    loss = dict(
        type = "MultiTaskLoss",
        # weight = [1.0, 2.0, 1.0],
        # reduction = "mean"
    ),
    # 优化器
    optimizer = dict(
        type = "AdamW",
        lr = 0.00001,
        betas = (0.9, 0.999),  # 一阶和二阶矩估计的指数衰减率
        eps = 1e-08,  # 增加数值稳定性的小常数
        weight_decay = 0.01
    )
)