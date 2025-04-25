# ailab
深度学习模型训练流


ailab/
├── configs/
│   └── train.yaml              # 全流程配置：支持分布式、AMP、日志等
├── mmcv/
│   └── __init__.py             # 可选 MMDet/MMCV 接口适配
├── models/
│   ├── __init__.py             # models 包
│   └── resnet50.py             # 用户自定义 ResNet50
├── src/
│   ├── registry.py             # Registry 与模块注册
│   ├── builder.py              # 构建器：datasets/models/optimizers
│   ├── workflow.py             # WorkFlow：训练/验证/测试/分布式调度
│   ├── hooks.py                # Hook：日志、断点、LR、分布式、AMP
│   ├── hooks_extra.py          # Hook：TensorBoard、WandB
│   └── utils/
│       ├── logging.py          # 日志管理
│       ├── checkpoint.py       # 检查点存取
│       ├── lr_scheduler.py     # 学习率调度器
│       ├── inference.py        # 推理接口
│       └── dist_utils.py       # 分布式初始化 & Sampler
├── tools/
│   ├── train.py                # 训练脚本（支持单/多卡、AMP）
│   ├── test.py                 # 验证脚本
│   └── infer.py                # 推理脚本
└── requirements.txt            # 依赖：torch, torchvision, mpi4py, wandb, tensorboard