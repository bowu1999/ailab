# ailab
深度学习模型训练流

## 概览
本仓库实现了一个轻量级的、借鉴 OpenMMLab 设计思路的深度学习框架（ailab），支持从数据集注册、模型构建，到训练（含 DDP 与 AMP）、日志记录（控制台、TensorBoard、Weights & Biases）、断点续训、评价指标、推理部署的全流程。用户只需编辑一个 YAML 配置文件，训练和推理脚本即可自动加载所有组件。

## 特性
模块化注册表：统一管理数据集、模型、优化器和指标

配置驱动：一个 YAML 文件定义数据、模型、优化器、Hook、训练流程与分布式设置

分布式训练：基于 NCCL 后端的 DDP，一键设置 torch.cuda.set_device 并在 barrier() 中指定 device_ids

混合精度：可选 AMP（自动混合精度）支持

Hook 系统：日志、断点续训、学习率调度、指标计算、TensorBoard & W&B 集成

进度条和样本计数：使用 tqdm（仅 Rank 0 显示）

断点续训：自动从检查点恢复，并支持动态学习率策略

依赖：
```bash
Python ≥3.7

PyTorch ≥1.9（含 CUDA）

torchvision

mpi4py（用于分布式启动）

wandb、tensorboard
```
## 项目结构
```bash
ailab/
├── configs/
│   └── train.yaml              # 全流程配置：支持分布式、AMP、日志等
├── src/ailab
|   │   ├── registry.py             # Registry 与模块注册
|   │   ├── builder.py              # 构建器：datasets/models/optimizers
|   │   ├── workflow.py             # WorkFlow：训练/验证/测试/分布式调度
|   │   ├── hooks.py                # Hook：日志、断点、LR、分布式、AMP
|   │   ├── hooks_extra.py          # Hook：TensorBoard、WandB
|   │   └── utils/
|   │   │   ├── logging.py          # 日志管理
|   │   │   ├── checkpoint.py       # 检查点存取
|   │   │   ├── lr_scheduler.py     # 学习率调度器
|   │   │   ├── inference.py        # 推理接口
|   │   │   └── dist_utils.py       # 分布式初始化 & Sampler
|   │   ├── datasets
|   │   │   ├── __init__.py             # dataset 包
|   │   │   └── base
|   │   │   │   ├── __init__.py         # base 包
|   │   │   │   └── _base.py            # 各种数据集接口
|   │   │   │   └── cv_base_dataset.py  # 基于特定任务的数据集
|   │   │   └── utils
|   │   │   │   ├── __init__.py         # utils 包
|   │   │   │   └── **.py               # 各种数据集需要的工具库
|   │   │   └── custom_data.py          # 用户自定义数据集
|   │   ├── models/
|   │   │   ├── __init__.py             # models 包
|   │   │   └── base                    # 模型基类
|   │   │   │   ├── __init__.py         # base 包
|   │   │   │   └── resnet.py          # ResNet 模版
|   │   │   │   └── **.py               # 各种模型接口
|   │   │   └── resnet50.py             # 用户自定义 ResNet50
|   │   │── metrics/
|   │   │   ├── __init__.py             # metrics 包
|   │   │   └── base                    # 基类
|   │   │   │   ├── __init__.py         # base 包
|   │   │   │   └── **.py               # 各种 Metrics 接口
│   │   |   └── **.py                   # 各种 Metrics
|   │   ├── losses/
|   │   │   ├── __init__.py             # losses 包
|   │   │   └── base                    # 基类
|   │   │   │   ├── __init__.py         # base 包
|   │   │   │   └── **.py               # 各种 Losses 接口
│   │   |   └── **.py                   # 各种 Losses
|   |   ├── tools/
|   │   |   ├── train.py                # 训练脚本（支持单/多卡、AMP）
|   |   │   ├── test.py                 # 验证脚本
│   |   |   └── infer.py                # 推理脚本
└── requirements.txt            # 依赖：torch, torchvision, mpi4py, wandb, tensorboard
```
## 快速开始
  您可以将 `ailab` 作为一个独立的库来使用，无需修改其源码。通过注册您自定义的模型、数据集和损失函数，并编写相应的配置文件，您可以灵活地构建和训练自己的深度学习模型。

以下是一个简洁的教程，指导您如何使用 `ailab` 进行模型训练：
> 源码仓库中的 example 中是一个简单的使用示例

---

## 源码安装 AILab

将 `ailab` 源码克隆到本地，并安装依赖：

```bash
git clone https://github.com/bowu1999/ailab.git
cd ailab
pip install -e .
```

## 创建并注册自定义组件

在一个新的 Python 文件（例如 `train.py`）中，定义并注册您的自定义模型、数据集、损失函数等。


```python
# train.py
from ailab import ailab_train
from ailab.registry import MODELS, DATASETS, LOSSES

from config import cfg

@MODELS.register_module()
class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 初始化模型

    def forward(self, x):
        # 定义前向传播
        return x

@DATASETS.register_module()
class MyDataset(Dataset):
    def __init__(self, ...):
        # 初始化数据集

    def __getitem__(self, idx):
        # 获取数据项
        return data

    def __len__(self):
        # 返回数据集大小
        return size

@LOSSES.register_module()
class MyLoss(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 初始化损失函数

    def forward(self, output, target):
        # 计算损失
        return loss

if __name__ == '__main__':
    ailab_train(cfg)
```

## 编写配置文件

创建一个配置文件（例如 `config.py`），指定训练的各项参数。

```python
# config.py
cfg = dict(
    seed=42,
    work_dir="./work_dir",
    dist=dict(
        backend="nccl",
        init_method="env://",
    ),
    model=dict(
        type="MyModel",
        # 其他模型参数
    ),
    data=dict(
        train=dict(
            type="MyDataset",
            # 其他数据集参数
        ),
        val=dict(
            type="MyDataset",
            # 其他数据集参数
        ),
        train_dataloader=dict(
            batch_size=32,
            num_workers=4,
        ),
        val_dataloader=dict(
            batch_size=32,
            num_workers=4,
        ),
    ),
    loss=dict(
        type="MyLoss",
        # 其他损失函数参数
    ),
    optimizer=dict(
        type="Adam",
        lr=0.001,
    ),
    total_epochs=100,
    hooks=dict(
        logger=dict(
            type="LoggerHook",
            interval=10,
            log_dir="./logs",
        ),
    ),
)
```


## 启动训练

使用以下命令启动分布式训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=12345 train.py
```


确保在配置文件中设置了正确的分布式训练参数，并根据您的硬件环境调整相关设置。

---

通过上述步骤，您可以灵活地使用 AILab 进行自定义模型的训练，而无需修改其源码。

查看日志：

控制台：Rank 0 打印进度、Loss、指标

TensorBoard：日志保存在 <work_dir>/tf_logs

W&B（如启用）：在线 Dashboard

断点续训：在配置文件中设置 resume_from 或自动从最新检查点恢复


# 配置文件结构（configs/train.py）

配置文件 `config.py` 以字典格式定义，主要包含以下部分：

### 【基础配置】

| 参数 | 说明 |
|:----|:----|
| `seed` | 随机种子，保证结果可复现 |
| `work_dir` | 工作目录，用于保存日志、模型权重等 |
| `resume_from` | （可选）从某个 checkpoint 恢复训练 |

---

### 【分布式训练设置】

| 参数 | 说明 |
|:----|:----|
| `dist.backend` | 通常设为 `"nccl"`（NVIDIA GPU 通讯） |
| `dist.init_method` | 初始化方式，默认 `"env://"` |

无需手动指定 `world_size`，由 `torchrun --nproc_per_node` 自动设置。

---

### 【训练流程控制】

| 参数 | 说明 |
|:----|:----|
| `workflow` | 训练/验证阶段及其迭代次数 |
| `total_epochs` | 总训练轮数 |

---

### 【Hooks 配置】

包含一系列自动化钩子，例如：

- `metrics`：自动计算 top1、top5 准确率
- `checkpoint`：保存模型
- `resume`：支持断点续训
- `lr_scheduler`：学习率调度器（如 `CosineAnnealingLR`、`StepLR`）
- `logger`：控制台+文件日志
- `tensorboard`：日志可视化
- `wandb`：可选接入 W&B 项目管理（可关闭）

---

### 【数据集与Dataloader】

| 参数 | 说明 |
|:----|:----|
| `data.train` | 训练集设置，需指定 `type` 和标注文件路径 |
| `data.val` | 验证集设置 |
| `train_dataloader` | batch size、workers 等参数 |
| `val_dataloader` | 同上 |

**注意：** 目前使用的是 `ClassificationImageDataset`，自带图像分类的标准数据集读取方式。

---

### 【模型定义】

| 参数 | 说明 |
|:----|:----|
| `model.type` | 如 `"resnet50"`，支持框架内置模型或注册自定义模型 |
| `num_classes` | 类别数 |

---

### 【损失函数】

| 参数 | 说明 |
|:----|:----|
| `loss.type` | 损失函数类型，如 `"CrossEntropyLoss"` |
| `loss.weight` | （可选）各类别权重 |
| `loss.reduction` | 损失聚合方式，如 `"mean"` |

---

### 【优化器】

| 参数 | 说明 |
|:----|:----|
| `optimizer.type` | 如 `"AdamW"`、`"Adam"` 等 |
| `lr` | 初始学习率 |
| `weight_decay` | 权重衰减正则项 |

---

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