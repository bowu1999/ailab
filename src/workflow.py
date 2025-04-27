import os
import torch
import random
import numpy as np
from tqdm import tqdm

from src.builder import build_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WorkFlow:
    def __init__(self, cfg, model, optimizer, criterion):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        # 工作目录
        self.work_dir = cfg['work_dir']
        os.makedirs(self.work_dir, exist_ok=True)
        # 构建 metrics
        self.metrics = build_metrics(cfg.get('metrics', {}))
        self.metric_results = {}
        # state
        self.hooks = []
        self.epoch = 0
        self.iter = 0
        self.last_loss = None
        self.last_outputs = None
        self.last_targets = None

    def _get_world_size(self):
        # 优先用 torch.distributed，fallback 到 cfg['dist']
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()

        return self.cfg.get('dist', {}).get('world_size', 1)

    def _call_model(self, data):
        # 解析 inputs/targets
        if isinstance(data, dict):
            inputs = data.get('input')
            targets = {k: v for k, v in data.items() if k != 'input'}
            if len(targets) == 1:
                targets = next(iter(targets.values()))
        elif isinstance(data, (list, tuple)) and len(data) >= 2:
            inputs, targets = data[0], data[1]
        else:
            inputs, targets = data, None
        # 将 inputs 和 targets 移动到模型所在的设备上
        device = next(self.model.parameters()).device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device, non_blocking=True)
        if isinstance(targets, torch.Tensor):
            targets = targets.to(device, non_blocking=True)
        outputs = self.model(inputs)
        self.last_outputs = outputs
        self.last_targets = targets

        return outputs, targets

    def register_hooks(self, hooks):
        self.hooks.extend(hooks)

    def call_hook(self, name):
        for hook in self.hooks:
            getattr(hook, name, lambda wf: None)(self)

    def run(self, data_loaders, workflow_cfg, total_epochs):
        self.call_hook('before_run')
        for ep in range(self.epoch, total_epochs):
            self.epoch = ep
            set_seed(self.cfg.get('seed', 0) + ep)
            for phase_cfg in workflow_cfg:
                phase = phase_cfg['phase']
                iters = phase_cfg.get('iters', None)
                self.call_hook(f'before_{phase}_epoch')
                method = getattr(self, f'_{phase}_epoch', None)
                if method is not None:
                    method(data_loaders.get(phase, []), iters)
                self.call_hook(f'after_{phase}_epoch')
        self.call_hook('after_run')

    def _train_epoch(self, loader, iters):
        self.model.train()
        world_size = self._get_world_size()
        total_iters = iters if iters is not None else len(loader)
        self.max_iter = total_iters

        is_main = int(os.environ.get('LOCAL_RANK', 0)) == 0
        pbar = tqdm(total=total_iters,
                    desc=f'Train Epoch {self.epoch}',
                    disable=not is_main)
        samples = 0
        data_iter = iter(loader)

        try:
            for _ in range(total_iters):
                self.call_hook('before_train_iter')

                try:
                    batch = next(data_iter)
                except StopIteration:
                    break

                outputs, targets = self._call_model(batch)
                if targets is None:
                    raise ValueError('Training data must return targets')

                # 确保 outputs 和 targets 在同一设备上
                device = outputs.device
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(device)

                loss = self.criterion(outputs, targets)
                self.last_loss = loss.item() if hasattr(loss, 'item') else loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.iter += 1

                # 累计样本数
                bs = targets.size(0) if isinstance(targets, torch.Tensor) else targets[0].size(0)
                samples += bs * world_size

                if is_main:
                    pbar.set_postfix({'Samples': samples})
                self.call_hook('after_train_iter')
                pbar.update(1)
        finally:
            pbar.close()

    def _val_epoch(self, loader, iters=None):
        self.model.eval()
        world_size = self._get_world_size()
        total_iters = iters if iters is not None else len(loader)
        self.max_iter = total_iters
        is_main = int(os.environ.get('LOCAL_RANK', 0)) == 0
        pbar = tqdm(total=total_iters,
                    desc=f'Val   Epoch {self.epoch}',
                    disable=not is_main)
        samples = 0
        data_iter = iter(loader)
        # before_val_epoch Hook（比如 MetricHook 会 reset）
        self.call_hook('before_val_epoch')
        try:
            with torch.no_grad():
                for _ in range(total_iters):
                    self.call_hook('before_val_iter')
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break
                    outputs, targets = self._call_model(batch)
                    # 累计样本数（targets 可能是 dict/tuple，需要自行调整）
                    if targets is not None:
                        bs = targets.size(0) if isinstance(targets, torch.Tensor) else targets[0].size(0)
                        samples += bs * world_size
                    if is_main:
                        pbar.set_postfix({'Samples': samples})
                    # after_val_iter Hook（MetricHook 会 update）
                    self.call_hook('after_val_iter')
                    pbar.update(1)
        finally:
            pbar.close()
        # after_val_epoch Hook（MetricHook 会 compute 并写入 wf.metric_results）
        self.call_hook('after_val_epoch')

    def _test_epoch(self, loader, iters=None):
        # test 完全沿用 val 的逻辑，只是 Hook 名称不同
        self.model.eval()
        world_size = self._get_world_size()
        total_iters = iters if iters is not None else len(loader)
        self.max_iter = total_iters
        is_main = int(os.environ.get('LOCAL_RANK', 0)) == 0
        pbar = tqdm(total=total_iters,
                    desc=f'Test  Epoch {self.epoch}',
                    disable=not is_main)
        samples = 0
        data_iter = iter(loader)
        self.call_hook('before_test_epoch')
        try:
            with torch.no_grad():
                for _ in range(total_iters):
                    self.call_hook('before_test_iter')
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break
                    outputs, targets = self._call_model(batch)
                    if targets is not None:
                        bs = targets.size(0) if isinstance(targets, torch.Tensor) else targets[0].size(0)
                        samples += bs * world_size
                    if is_main:
                        pbar.set_postfix({'Samples': samples})
                    self.call_hook('after_test_iter')
                    pbar.update(1)
        finally:
            pbar.close()
        self.call_hook('after_test_epoch')