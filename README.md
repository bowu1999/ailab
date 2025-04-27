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

## 安装
```bash
git clone https://github.com/yourusername/ailab.git
cd ailab
pip install -r requirements.txt
```
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
├── configs/            # YAML 配置文件
│   └── train.yaml
├── datasets/           # 数据集类及工具
├── metric/             # 指标实现
├── models/             # 模型定义（ResNet 模板 + 用户模型）
├── src/
│   ├── registry.py     # 全局注册表
│   ├── builder.py      # 构建器：dataset/model/optimizer/metric
│   ├── workflow.py     # WorkFlow：训练/验证/测试/分布式调度
│   ├── hooks.py        # 核心 Hook
│   ├── hooks_extra.py  # TensorBoard & W&B Hook
│   └── utils/          # 日志、检查点、LR 调度、分布式工具、推理
├── tools/
│   ├── train.py        # 训练入口脚本（支持 –config）
│   ├── test.py         # 验证脚本
│   └── infer.py        # 推理脚本
└── requirements.txt
```
## 快速开始
编辑配置：复制并修改 configs/train.yaml

启动训练（单/多卡均可）：

```bash
# 多卡（4 GPU）：
torchrun --nproc_per_node=4 tools/train.py --config configs/train.yaml
```
查看日志：

控制台：Rank 0 打印进度、Loss、指标

TensorBoard：日志保存在 <work_dir>/tf_logs

W&B（如启用）：在线 Dashboard

断点续训：在配置文件中设置 resume_from 或自动从最新检查点恢复

配置说明
以下逐项说明 configs/train.yaml 中各字段含义：

```yaml
seed: 42                       # 随机种子
work_dir: /path/to/ailab-workspace  # 工作目录
resume_from: null              # 从该路径恢复模型
dist:
  backend: nccl                # 分布式后端
  world_size: 4                # 进程总数
  init_method: env://          # 初始化方法（用于 torchrun）  
amp:
  enabled: true                # 是否启用混合精度
  opt_level: O1                # AMP 优化等级（兼容 Apex 旧接口）

workflow:
  - { phase: train, iters: 4 }  # 流程阶段（train/val/test），iters=None 表示遍历整个数据集
  - { phase: val, iters: 4 }

total_epochs: 20               # 总训练轮数

hooks:
  metrics:                     # 指标计算 Hook
    type: MetricHook
    top1:
      type: "Accuracy"         # Accuracy 指标，topk=1
      topk: 1
    top5:
      type: "Accuracy"         # Accuracy 指标，topk=5
      topk: 5

  checkpoint:                   # 检查点 Hook
    type: CheckpointHook
    interval: 1                 # 每多少 epoch 保存一次

  resume:
    type: ResumeHook
    enable: true                # 如果设置了 resume_from，则自动加载

  lr_scheduler:
    type: LrSchedulerHook       # 学习率调度 Hook
    scheduler:
      type: StepLR
      step_size: 10
      gamma: 0.5

  logger:                       # 日志打印 Hook
    type: LoggerHook
    interval: 1                 # 每多少 iterate 打印一次
    log_dir: ${work_dir}/logs   # 日志目录（需支持变量插值）
    log_items:
      lr: "optimizer.param_groups.0.lr"
      top1: "meters['top1'].avg"

  ddp:
    type: DDPHook               # 分布式数据并行 Hook

  amp:
    type: AMPHook               # 混合精度 Hook

  tensorboard:
    type: TensorboardHook       # TensorBoard Hook，写入 `<work_dir>/tf_logs`

  wandb:
    type: WandbHook             # Weights & Biases Hook
    init_args:
      project: my_project
      name: ${experiment_name}  # 需支持变量插值
      mode: disabled            # 'disabled' 或 'offline' 模式

data:
  train:
    type: ClassificationImageDataset
    annotation_file: /path/to/train.jsonl
    x_key: image_path
    y_key: category_id

  val:
    type: ClassificationImageDataset
    annotation_file: /path/to/val.jsonl
    x_key: image_path
    y_key: category_id

  train_dataloader:
    batch_size: 8
    shuffle: true

  val_dataloader:
    batch_size: 8
    shuffle: false

model:
  type: resnet50               # 从 models/resnet50.py 加载
  num_classes: 12
  # pretrained: true           # 是否加载预训练权重

optimizer:
  type: Adam
  lr: 0.001
  weight_decay: 0.0001

lr_scheduler:
  type: StepLR
  step_size: 10
  gamma: 0.5
```
关于变量插值

${work_dir}、${experiment_name} 等占位符需要使用支持插值的解析器（如 OmegaConf）或在加载前后自行替换。

贡献指南
Fork 本仓库并 clone

创建分支 (git checkout -b feature/xxx)

提交代码 (git commit -am 'Add feature')

推送并发起 Pull Request

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