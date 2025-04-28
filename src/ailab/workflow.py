import os
import torch
import random
import numpy as np
from tqdm import tqdm

from ailab.builder import build_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_output_device(outputs):
    """
    从模型输出outputs中提取device，支持Tensor、tuple、list、dict等各种类型
    如果没找到Tensor，则返回 cpu device。
    """
    if torch.is_tensor(outputs):
        return outputs.device
    elif isinstance(outputs, (tuple, list)):
        # 递归查找第一个Tensor
        for item in outputs:
            device = get_output_device(item)
            if device is not None:
                return device
    elif isinstance(outputs, dict):
        for v in outputs.values():
            device = get_output_device(v)
            if device is not None:
                return device
    # 没有找到tensor时的fallback
    return torch.device('cpu')


def move_to_device(data, device):
    """
    递归将Tensor转移到device，支持Tensor、dict、list、tuple等结构。
    其它类型原样返回。
    """
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(v, device) for v in data)
    else:
        return data


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


def to_float_recursively(data):
    if torch.is_tensor(data):
        return data.float()
    elif isinstance(data, dict):
        return {k: to_float_recursively(v) for k,v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_float_recursively(v) for v in data)
    else:
        return data


def to_correct_dtype_targets(targets, classify_keys=['cls']):
    """
    对 targets 递归处理，
    - 如果 key 属于 classify_keys，转为long（分类标签）
    - 其它转 float（浮点输出）
    """
    if torch.is_tensor(targets):
        # 如果单个 tensor，没有键，默认转 long？
        return targets.long()
    elif isinstance(targets, dict):
        ret = {}
        for k, v in targets.items():
            if k in classify_keys and torch.is_tensor(v):
                ret[k] = v.long()
            elif torch.is_tensor(v):
                ret[k] = v.float()
            else:
                ret[k] = v
        return ret
    elif isinstance(targets, (list, tuple)):
        return type(targets)(to_correct_dtype_targets(v, classify_keys) for v in targets)
    else:
        return targets


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
        if isinstance(data, dict):
            inputs = data.get('input')
            targets = {k: v for k, v in data.items() if k != 'input'}
            if len(targets) == 1:
                targets = next(iter(targets.values()))
        elif isinstance(data, (list, tuple)) and len(data) >= 2:
            inputs, targets = data[0], data[1]
        else:
            inputs, targets = data, None

        device = next(self.model.parameters()).device
        inputs = move_to_device(inputs, device)
        if torch.is_tensor(inputs):
            inputs = inputs.float()
        else:
            inputs = to_float_recursively(inputs)

        if targets is not None:
            targets = move_to_device(targets, device)
            targets = to_correct_dtype_targets(targets, classify_keys=['cls'])

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
                device = get_output_device(outputs)
                loss = self.criterion(outputs, targets)
                self.last_loss = loss.item() if hasattr(loss, 'item') else loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.iter += 1
                # 累计样本数
                bs = get_batch_size(targets)
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
                        bs = get_batch_size(targets)
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
                        bs = get_batch_size(targets)
                        samples += bs * world_size
                    if is_main:
                        pbar.set_postfix({'Samples': samples})
                    self.call_hook('after_test_iter')
                    pbar.update(1)
        finally:
            pbar.close()
        self.call_hook('after_test_epoch')