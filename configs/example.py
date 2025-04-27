# configs/train.py

cfg = dict(
    # 基础设置
    seed          = 42,
    work_dir      = "/mnt/.../ailab-workspace",
    resume_from   = None,

    # 分布式
    dist = dict(
        backend     = "nccl",
        world_size  = 4,
        init_method = "env://",
    ),

    # 混合精度
    amp = dict(
        enabled  = True,
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
            top1  = dict(type="Accuracy", topk=1),
            top5  = dict(type="Accuracy", topk=5),
        ),
        checkpoint = dict(type="CheckpointHook", interval=1),
        resume     = dict(type="ResumeHook", enable=True),
        lr_scheduler = dict(
            type      = "LrSchedulerHook",
            scheduler = dict(type="StepLR", step_size=10, gamma=0.5),
        ),
        logger     = dict(
            type     = "LoggerHook",
            interval = 1,
            log_dir  = "{work_dir}/logs".format(work_dir="/mnt/.../ailab-workspace"),
            log_items = dict(
                lr   = "optimizer.param_groups.0.lr",
                top1 = "meters['top1'].avg",
            ),
        ),
        ddp        = dict(type="DDPHook"),
        amp        = dict(type="AMPHook"),
        tensorboard = dict(type="TensorboardHook"),
        wandb      = dict(
            type      = "WandbHook",
            init_args = dict(project="my_project",
                             name="experiment_name",
                             mode="disabled"),
        ),
    ),

    # 数据
    data = dict(
        train = dict(
            type            = "ClassificationImageDataset",
            annotation_file = "...@train.jsonl",
            x_key           = "image_path",
            y_key           = "category_id",
        ),
        val   = dict(
            type            = "ClassificationImageDataset",
            annotation_file = "...@val.jsonl",
            x_key           = "image_path",
            y_key           = "category_id",
        ),
        train_dataloader = dict(batch_size=8, shuffle=True),
        val_dataloader   = dict(batch_size=8, shuffle=False),
    ),

    # 模型
    model = dict(type="resnet50", num_classes=12),

    # 优化器
    optimizer = dict(type="Adam", lr=0.001, weight_decay=1e-4),

    # 学习率调度
    lr_scheduler = dict(type="StepLR", step_size=10, gamma=0.5),
)
