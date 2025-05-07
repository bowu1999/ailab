import os
import torch
import random
import numpy as np
from tqdm import tqdm

from ailab.builder import build_metrics
from ailab.utils import call_fn, OutputWrapper


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_output_device(outputs):
    if torch.is_tensor(outputs):
        return outputs.device
    elif isinstance(outputs, (tuple, list)):
        for item in outputs:
            device = get_output_device(item)
            if device is not None:
                return device
    elif isinstance(outputs, dict):
        for v in outputs.values():
            device = get_output_device(v)
            if device is not None:
                return device
    return torch.device('cpu')


def get_batch_size(targets):
    if torch.is_tensor(targets):
        return targets.size(0)
    elif isinstance(targets, dict):
        for v in targets.values():
            if torch.is_tensor(v):
                return v.size(0)
    elif isinstance(targets, (list, tuple)):
        for v in targets:
            if torch.is_tensor(v):
                return v.size(0)
    raise ValueError("Cannot infer batch size from targets")


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(v, device) for v in data)
    else:
        return data


class WorkFlow:
    """
    通用训练/验证/测试流程类 (WorkFlow):

    核心职责：
      1. 根据 cfg 实例化并包装模型、优化器、Loss 及 Hooks 等组件
      2. 在各阶段 (train/val/test) 调度数据读取、前向、Loss 计算及 Hook 调用
      3. 支持 Hook 机制，所有指标的 reset/update/compute 均由 MetricHook 处理

    使用示例：
      # 1. 实例化核心组件
      model     = build_from_cfg(cfg['model'], registry=MODEL_REGISTRY)
      optimizer = build_from_cfg(cfg['optimizer'], registry=OPTIMIZER_REGISTRY)
      criterion = build_from_cfg(cfg['loss'], registry=LOSS_REGISTRY)

      # 2. 构造 DataLoader
      train_ds    = build_from_cfg(cfg['data']['train'], registry=DATASET_REGISTRY)
      train_loader= DataLoader(train_ds, **cfg['data']['train_dataloader'])
      val_ds      = build_from_cfg(cfg['data']['val'], registry=DATASET_REGISTRY)
      val_loader  = DataLoader(val_ds, **cfg['data']['val_dataloader'])

      # 3. 初始化 WorkFlow 并注册所有 Hook
      wf = WorkFlow(cfg, model, optimizer, criterion)
      hooks = build_from_cfg_list(cfg['hooks'], registry=HOOKS_REGISTRY)
      wf.register_hooks(hooks)

      # 4. 执行
      wf.run({'train':train_loader, 'val':val_loader}, cfg['workflow'], cfg['total_epochs'])
    """
    def __init__(self, cfg, model, optimizer, criterion):
        self.cfg = cfg
        # 模型包装：规范输出为 dict
        self.model_mapping = cfg.get('model', {}).get('mapping', {})
        self.model = model
        # 优化器与 Loss
        self.optimizer = optimizer
        self.criterion = criterion
        # 工作目录
        self.work_dir = cfg['work_dir']
        os.makedirs(self.work_dir, exist_ok=True)
        # Hook 存储
        self.hooks = []
        # 训练状态
        self.epoch = 0
        self.iter = 0
        self.last_loss = None
        self.last_outputs = None
        self.last_targets = None

    def register_hooks(self, hooks):
        """注册 Hook 列表"""
        self.hooks.extend(hooks)

    def call_hook(self, name):
        """调用所有 Hook 的 name 方法"""
        for hook in self.hooks:
            getattr(hook, name, lambda wf: None)(self)

    def _get_world_size(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return self.cfg.get('dist', {}).get('world_size', 1)

    def _call_model(self, batch):
        """
        1) 支持 batch: dict OR tuple/list
        2) 自动把 (inputs,targets) 包装为 {'input':..., 'target':...}
        3) 其余逻辑不变
        """
        # —— 新增：tuple 支持 —— #
        if not isinstance(batch, dict):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # 假设第 0 项是 inputs，第 1 项是 targets
                batch = {'input': batch[0], 'target': batch[1]}
            else:
                raise ValueError(f"Unsupported batch type {type(batch)}")

        # 找到底层模型（兼容 DDP）
        wrapped = getattr(self.model, 'module', self.model)
        device  = next(wrapped.parameters()).device

        # 1) 设备迁移
        batch = move_to_device(batch, device)

        # 2) 模型前向：只传入模型真正需要的字段
        outputs = call_fn(self.model, batch, mapping=self.model_mapping)
        self.last_outputs = outputs

        # 3) 合并字典，供 Loss/Metric 调用
        self.last_data = {**batch, **outputs}
        return outputs, self.last_data

    def run(self, data_loaders, workflow_cfg, total_epochs):
        """
        执行流程：依次触发 before_run -> 各 phase -> after_run
        """
        self.call_hook('before_run')
        for ep in range(self.epoch, total_epochs):
            self.epoch = ep
            set_seed(self.cfg.get('seed', 0) + ep)
            for phase_cfg in workflow_cfg:
                phase = phase_cfg['phase']  # 'train'/'val'/'test'
                iters = phase_cfg.get('iters', None)
                self.call_hook(f'before_{phase}_epoch')
                method = getattr(self, f'_{phase}_epoch', None)
                if method:
                    method(data_loaders.get(phase, []), iters)
                self.call_hook(f'after_{phase}_epoch')
        self.call_hook('after_run')

    def _train_epoch(self, loader, iters):
        self.model.train()
        world_size = self._get_world_size()
        total_iters = iters or len(loader)
        self.max_iter = total_iters
        is_main = (int(os.environ.get('LOCAL_RANK', 0)) == 0)
        pbar = tqdm(total=total_iters, desc=f'Train {self.epoch}', disable=not is_main, leave=False)
        samples = 0
        data_iter = iter(loader)
        try:
            for _ in range(total_iters):
                self.call_hook('before_train_iter')
                cpu_batch = next(data_iter, None)
                if cpu_batch is None:
                    break
                # 让 _call_model 同时返回 outputs 和 GPU 上的 data
                outputs, data = self._call_model(cpu_batch)
                # 计算 Loss：全部从 data（GPU 上）读取
                for k, v in data.items():
                    if torch.is_tensor(v) and v.dtype == torch.float64:
                        data[k] = v.float()
                loss = call_fn(self.criterion, data, mapping = self.cfg['loss'].get('mapping', {}))
                self.last_loss = loss.item() if hasattr(loss, 'item') else loss
                # 反向 & 优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 样本统计
                bs = get_batch_size(self.last_data)
                samples += bs * world_size
                if is_main:
                    pbar.set_postfix({'Samples': samples})
                self.iter += 1
                self.call_hook('after_train_iter')
                pbar.update(1)
        finally:
            pbar.close()

    def _val_epoch(self, loader, iters=None):
        self.model.eval()
        world_size = self._get_world_size()
        total_iters = iters or len(loader)
        self.max_iter = total_iters
        is_main = (int(os.environ.get('LOCAL_RANK', 0)) == 0)
        pbar = tqdm(total=total_iters, desc=f'Val {self.epoch}', disable=not is_main, leave=False)
        samples = 0
        data_iter = iter(loader)
        try:
            with torch.no_grad():
                for _ in range(total_iters):
                    self.call_hook('before_val_iter')
                    batch = next(data_iter, None)
                    if batch is None:
                        break
                    outputs, data = self._call_model(batch)
                    loss = call_fn(
                        self.criterion,
                        data,
                        mapping = self.cfg.get('loss', {}).get('mapping', {})
                    )
                    # 记录到 last_loss，供 LoggerHook 打印
                    self.last_loss = loss.item() if hasattr(loss, 'item') else loss
                    # 触发 after_val_iter 钩子，MetricHook 会在此更新指标
                    self.call_hook('after_val_iter')
                    bs = get_batch_size(self.last_data)
                    samples += bs * world_size
                    if is_main:
                        pbar.set_postfix({'Samples': samples})
                    pbar.update(1)
        finally:
            pbar.close()

    def _test_epoch(self, loader, iters=None):
        self.model.eval()
        world_size = self._get_world_size()
        total_iters = iters or len(loader)
        self.max_iter = total_iters
        is_main = (int(os.environ.get('LOCAL_RANK', 0)) == 0)
        pbar = tqdm(total=total_iters, desc=f'Test {self.epoch}', disable=not is_main, leave=False)
        samples = 0
        data_iter = iter(loader)
        try:
            with torch.no_grad():
                for _ in range(total_iters):
                    self.call_hook('before_test_iter')
                    batch = next(data_iter, None)
                    if batch is None:
                        break
                    outputs, data = self._call_model(batch)
                    self.call_hook('after_test_iter')
                    bs = get_batch_size(self.last_data)
                    samples += bs * world_size
                    if is_main:
                        pbar.set_postfix({'Samples': samples})
                    pbar.update(1)
        finally:
            pbar.close()