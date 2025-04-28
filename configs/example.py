# configs/train.py

cfg = dict(
    # 基础设置
    seed = 42,
    work_dir = "/mnt/.../ailab-workspace",
    resume_from = None,
    # 分布式
    dist = dict(
        backend = "nccl",
        world_size = 4,
        init_method = "env://",
    ),
    # 混合精度
    amp = dict(
        enabled = True,
        opt_level = "O1",
    ),
    # 训练流程
    workflow = [
        dict(phase="train", iters=4),
        dict(phase="val",   iters=4),
        # dict(phase="test"), 
    ],
    total_epochs = 20,
    # Hooks
    hooks = dict(
        metrics = dict(
            type = "MetricHook",
            top1 = dict(type="Accuracy", topk=1),
            top5 = dict(type="Accuracy", topk=5),
        ),
        checkpoint = dict(type="CheckpointHook", interval=1),
        resume = dict(type="ResumeHook", enable=True),
        lr_scheduler = dict(
            type = "LrSchedulerHook",
            scheduler = dict(type="StepLR", step_size=10, gamma=0.5),
        ),
        logger = dict(
            type = "LoggerHook",
            interval = 1,
            log_dir = "{work_dir}/logs".format(work_dir="/mnt/.../ailab-workspace"),
            log_items = dict(
                lr = "optimizer.param_groups.0.lr",
                top1 = "meters['top1'].avg",
            ),
        ),
        ddp = dict(type="DDPHook"),
        amp = dict(type="AMPHook"),
        tensorboard = dict(type="TensorboardHook"),
        wandb = dict(
            type = "WandbHook",
            init_args = dict(project="my_project",
                             name="experiment_name",
                             mode="disabled"),
        ),
    ),
    # 数据
    data = dict(
        train = dict(
            type = "ClassificationImageDataset",
            annotation_file = "...@train.jsonl",
            x_key = "image_path",
            y_key = "category_id",
        ),
        val = dict(
            type = "ClassificationImageDataset",
            annotation_file = "...@val.jsonl",
            x_key = "image_path",
            y_key = "category_id",
        ),
        train_dataloader = dict(batch_size=8, shuffle=True),
        val_dataloader = dict(batch_size=8, shuffle=False),
    ),
    # 模型
    model = dict(type="resnet50", num_classes=12),
    # 损失
    # 损失
    loss = dict(
        type = "CrossEntropyLoss",
        weight = [1.0, 2.0, 1.0],
        reduction = "mean"
    ),
    # 优化器
    optimizer = dict(type="Adam", lr=0.001, weight_decay=1e-4),
)



# configs/train.py

# 如果你希望在这里使用变量插值，可以先定义 work_dir，再在下面复用：
work_dir = ""
experiment_name = "experiment_1"  # 你可以根据需要修改

cfg = dict(
    # 随机种子
    seed = 42,
    # 工作目录
    work_dir = work_dir,
    # 从哪个 checkpoint 恢复训练
    resume_from = None,
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
        dict(phase = "train", iters = 100), # iters 表示每个epoch中有多少个迭代，可以根据需要修改，可以不设置
        dict(phase = "val"),
        # {"phase": "test"}, 
    ],
    total_epochs = 200,
    # 钩子 (Hooks)
    hooks = dict(
        # 指标钩子：自动统计 top1/top5
        metrics = dict(
            type = "MetricHook",
            top1 = dict(type="Accuracy", topk=1),
            top5 = dict(type="Accuracy", topk=5),
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
                top1 = "meters['top1'].avg",
                top5 = "meters['top5'].avg",
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
            type = "ClassificationImageDataset",
            annotation_file = "train.jsonl",
            x_key = "image_path",
            y_key = "category_id",
        ),
        val = dict(
            type = "ClassificationImageDataset",
            annotation_file = "val.jsonl",
            x_key = "image_path",
            y_key = "category_id",
        ),
        train_dataloader = dict(
            batch_size=16,
            num_workers=16, 
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        ),
        val_dataloader   = dict(
            batch_size=16,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        ),
    ),
    # 模型
    model = dict(
        type        = "resnet50",
        num_classes = 12,
        # pretrained   = False,
    ),
    # 损失
    loss = dict(
        type = "CrossEntropyLoss",
        # weight = [1.0, 2.0, 1.0],
        # reduction = "mean"
    ),
    # 优化器
    optimizer = dict(
        type = "AdamW",
        lr = 0.0001,
        betas = (0.9, 0.999),  # 一阶和二阶矩估计的指数衰减率
        eps = 1e-08,  # 增加数值稳定性的小常数
        weight_decay = 0.01
    )
)
