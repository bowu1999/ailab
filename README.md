# ailab
深度学习模型训练流

```bash
ailab/
├── configs/
│   └── train.yaml              # 全流程配置：支持分布式、AMP、日志等
├── datasets
│   ├── __init__.py             # dataset 包
│   └── base
│   │   ├── __init__.py         # base 包
│   │   └── _base.py            # 各种数据集接口
|   │   └── cv_base_dataset.py  # 基于特定任务的数据集
|   └── utils
│   │   ├── __init__.py         # utils 包
│   │   └── **.py               # 各种数据集需要的工具库
│   └── custom_data.py          # 用户自定义数据集
├── models/
│   ├── __init__.py             # models 包
│   └── base                    # 模型基类
│   │   ├── __init__.py         # base 包
│   │   └── resnet.py          # ResNet 模版
│   │   └── **.py               # 各种模型接口
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
```

# config 配置参数介绍
## 1. dist:
|参数|意义|典型取值及说明|
|--|--|--|
|backend|分布式通信后端，决定多进程间如何同步。|nccl（多GPU推荐），gloo（支持CPU）|
|world_size|进程总数，代表有多少份参与分布式训练。|1（单机/单卡），8（比如2机各4卡）|
|init_method|分布式初始化方式及主节点位置获取方式。|env://（推荐），tcp://ip:port|
- backend: nccl

含义：指定分布式通信后端

常见取值：nccl：NVIDIA Collective Communications Library。
最快且默认用于多GPU训练，只能在NVIDIA GPU环境下用。几乎所有多卡训练都推荐用它。

gloo：PyTorch自带的通用后端，支持CPU和GPU，但在多GPU上没有nccl高效。

mpi：基于MPI，适合对分布式有特别需求的场景。现在用得较少。

示例：
```python
dist.init_process_group(backend="nccl", ...)
```
- world_size: 1

含义：全局参与分布式训练的“总进程数”。

通常每张卡/每台机器一个进程。

你在单机单卡训练时一般设为1（其实可以不设，但合理填1最安全）。

多机多卡/单机多卡要填实际的总进程数。例如：2台机器，每台4张卡，world_size 就是8。

用途：


让每个分布式进程都知道总共有几份任务会同步参数/数据。

- init_method: env://

含义：分布式通信初始化的方法。

"env://" 代表：“通讯参数通过环境变量获得”。是目前官方最推荐的，理由是跨环境、易于启动脚本兼容容器/K8S任务。

你用 torchrun 或 python -m torch.distributed.launch 启动时，这些会自动设置 MASTER_ADDR、MASTER_PORT、RANK、WORLD_SIZE 等环境变量。

其它常见值：

tcp://IP:PORT：直接指定主节点IP和端口。（如tcp://127.0.0.1:23456）

file:///path/to/some/file：少见，现在基本不用，早期某些实验环境可用。