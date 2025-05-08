## 概述

AILab 是一个轻量级、配置驱动的深度学习训练框架，覆盖从数据集注册、模型构建，到分布式训练（DDP + NCCL）、混合精度（AMP）、日志（Console/TensorBoard/W&B）、断点续训、指标计算，以及推理部署的全流程。  
用户只需编写一个 Python 配置文件，即可对接自定义的数据集、模型与损失函数，无需改动框架源码。

---

## 核心特性

### 1. 模块化注册表  
- **Registry 机制**：统一管理数据集（`DATASETS`）、模型（`MODELS`）、损失（`LOSSES`）、指标（`METRICS`）、Hook（`HOOKS`）等组件。  
- **插件式扩展**：用户在任意位置定义新类，只需 `@XXX.register_module()` 一行即可纳入调度。

### 2. 配置驱动  
- **单一 Python 配置**：所有超参、路径、流程、钩子、分布式设置都写在一个 `cfg` 字典里。  
- **动态绑定**：`call_fn` 根据 `mapping` 自动将数据字典中的字段，映射到模型、Loss、Metric 的方法参数。

### 3. 分布式训练  
- 基于 NCCL 后端的 `torch.nn.parallel.DistributedDataParallel`（支持多机多卡）。  
- `torchrun` 一行命令启动，全局环境变量自动注入 `RANK`、`WORLD_SIZE`、`MASTER_ADDR/PORT`。

### 4. 混合精度加速  
- **AMP Hook**：可选开启，训练与验证自动在半精度下加速，保持数值稳定。

### 5. 灵活的 Hook 系统  
- **MetricHook**：支持多指标、阶段过滤（如仅在验证阶段计算 PSNR），并自动与 Logger/TensorBoard/W&B 集成。  
- **LoggerHook**：周期性输出训练状态、样本数和指标平均值。  
- **CheckpointHook / ResumeHook**：自动保存与恢复模型权重与训练状态。  
- **LrSchedulerHook**：集成多种调度器（CosineAnnealing、Warmup+Cosine 等）。  
- **DDPHook**：分布式 Barrier 与梯度同步优化。  
- **TensorboardHook & WandbHook**：一键接入可视化平台。

### 6. 鲁棒的数据管道  
- **统一抽象基类** (`LabBaseDataset`)：支持多种存储方式（单文件、文件夹分类、CSV/Excel、COCO/VOC 等）与子任务抽象（分类、回归、检测、分割、时序、音频、文本等）。  
- **自动容错**：全局打开 PIL 截断图像容错，`__getitem__` 捕获并跳过坏样本，保证 DataLoader 不因单张损坏文件卡死。  
- **tuple/dict 双兼容**：Dataset 可返回 `(inputs, targets)` 或 `{'input':…, 'target':…, …}`，Workflow 会自动包装 tuple→dict。

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/bowu1999/ailab.git
cd ailab
pip install -e .
```

### 2. 编写自定义组件

在自己的工作目录下新建 `train.py` 文件，并在其中进行如下注册：

```python
import torch.nn as nn
from torch.utils.data import Dataset
from ailab import ailab_train
from ailab.registry import MODELS, DATASETS, LOSSES

@MODELS.register_module()
class MyModel(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
    def forward(self, input):
        return self.linear(input)

@DATASETS.register_module()
class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = load_data(data_path)  # 任意加载方式
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y  # 支持 tuple，框架会自动包装

@LOSSES.register_module()
class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, output, target):
        return self.mse(output, target)

if __name__ == "__main__":
    from config import cfg
    ailab_train(cfg)
```

### 3. 编写配置文件 `config.py`

在同一工作目录下新建 `config.py` 文件：

```python
cfg = dict(
    seed = 42,
    work_dir = "./work_dir",
    dist = dict(backend="nccl", init_method="env://"),
    # 核心流程：train → val
    workflow = [{"phase":"train"},{"phase":"val"}],
    total_epochs = 50,
    data = dict(
        train = dict(type="MyDataset", data_path="train.pkl"),
        val   = dict(type="MyDataset", data_path="val.pkl"),
        train_dataloader = dict(batch_size=32, num_workers=4, pin_memory=True),
        val_dataloader   = dict(batch_size=64, num_workers=2),
    ),

    model = dict(
        type        = "MyModel",
        in_feats    = 128,
        out_feats   = 10,
        mapping     = {"input":"input"},   # forward(input=…)
        output_keys = ["output"],          # output Tensor 命名为 'output'
    ),

    loss = dict(
        type    = "MyLoss",
        mapping = {"output":"output", "target":"target"}
    ),

    optimizer = dict(type="AdamW", lr=1e-3, weight_decay=0.01),

    hooks = dict(
        metrics = dict(
            type = "MetricHook",
            top1 = dict(
                type    = "Top1Accuracy",
                mapping = {"pred":"output", "label":"target"},
                phases  = ["train","val"]
            ),
        ),
        logger = dict(
            type      = "LoggerHook",
            interval  = 100,
            log_dir   = "./work_dir/logs",
            log_items = dict(
                lr   = "optimizer.param_groups.0.lr",
                top1 = "meters['top1'].avg"
            )
        ),
        checkpoint  = dict(type="CheckpointHook", interval=1),
        resume      = dict(type="ResumeHook"),
        lr_scheduler= dict(type="LrSchedulerHook", scheduler=dict(type="CosineAnnealingLR", T_max=10)),
        ddp         = dict(type="DDPHook"),
        amp         = dict(type="AMPHook"),
        tensorboard = dict(type="TensorboardHook"),
        wandb       = dict(type="WandbHook", init_args=dict(project="my_proj", name="exp1")),
    )
)
```

### 启动训练
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=12345 train.py
```

> 源码仓库中的 example 中是一个简单的使用示例

---

## 关键点与提示

1. **映射 (`mapping`)**  
   - `cfg.model.mapping`：模型 `forward` 需要哪些字段映射到输入字典。  
   - `cfg.loss.mapping`、`cfg.hooks.metrics.*.mapping`：Loss/Metric 方法参数 → 数据字典字段。

2. **多输出**  
   - `output_keys` 定义模型返回 tuple 时各部分的命名。单输出可省略，框架默认 `["output"]`。

3. **阶段过滤**  
   - `phases` 字段控制 Metric 只在指定阶段生效（如只在验证时计算 PSNR）。

4. **自动容错**  
   - 坏文件会被跳过并打印警告；Pillow 截断图像容错已全局开启。

5. **tuple/dict 通用**  
   - Dataset 返回 `(x,y)` 或 `{'input':x,'target':y,'mask':...}` 均可，框架自动统一到 dict 处理。

6. **分布式与设备**  
   - Model、Loss、Metric 会自动 `.to(device)`；NCCL 后端下所有张量同步均在 GPU 上完成。

---

## 相关说明

ailab 中提供了函数 `remove_module_prefix`，用于移除 `checkpoint` 文件中模型相关参数的 `module.` 前缀。

在加载模型时可以使用如下的方式
```python
from ailab import remove_module_prefix

model = MyModel()
checkpoint_path = "/path/checkpoint.pth"
checkpoint = torch.load(checkpoint_path, weights_only=False)
model.load_state_dict(remove_module_prefix(checkpoint['model_state']))
```

通过上述说明，你可以快速上手 AILab，利用其高度模块化与配置驱动的设计，以最小改动实现复杂任务的训练与评估。
## 项目结构
```bash
ailab/
├── configs/
│   └── train.py                    # 全流程配置：支持分布式、AMP、日志等
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
|   │   │   │   └── resnet.py           # ResNet 模版
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