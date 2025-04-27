import os
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class WorkFlow:
    def __init__(self, cfg, model, optimizer, criterion):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.work_dir = cfg['work_dir']
        os.makedirs(self.work_dir, exist_ok=True)
        self.hooks = []
        self.epoch = 0
        self.iter = 0
        self.last_loss = None
        self.criterion = criterion
    
    def _call_model(self, data):
       """
       只把网络需要的“inputs”送进模型，返回 (outputs, targets) 格式：
       - 如果 data 是 dict，尝试从里边取 'img' 或 'input'；其余字段当 targets
       - 如果 data 是 (inputs, targets) tuple，分开处理
       - 其他格式当作 inputs，targets 置为 None
       """
       if isinstance(data, dict):
           # 可以根据 dataset 约定修改键名
           inputs = data.get('input')
           # 其他所有 key 都视作 targets 的不同组成部分
           targets = {k: v for k, v in data.items() if k != 'input'}
           # 如果只有一个 target，就直接拿出来
           if len(targets) == 1:
               targets = next(iter(targets.values()))
       elif isinstance(data, (list, tuple)) and len(data) >= 2:
           inputs, targets = data[0], data[1]
       else:
           inputs, targets = data, None
       outputs = self.model(inputs)

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
        data_iter = iter(loader)
        epoch_iters = iters if iters is not None else len(loader)
        for _ in range(epoch_iters):
            self.call_hook('before_train_iter')
            data = next(data_iter)
            outputs, targets = self._call_model(data)
            # 用 criterion 计算 loss
            if targets is None:
                raise ValueError('Training data must return targets for loss computation')
            # if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            #     print(f"targets: {targets} \n targets.type: {type(targets)}")
            loss = self.criterion(outputs, targets)
            self.last_loss = loss.item() if hasattr(loss, 'item') else loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iter += 1
            self.call_hook('after_train_iter')

    def _val_epoch(self, loader, iters=None):
        self.model.eval()
        with torch.no_grad():
            for idx, data in enumerate(loader):
               if iters is not None and idx >= iters:
                   break
               outputs, targets = self._call_model(data)

    def _test_epoch(self, loader, iters=None):
        self._val_epoch(loader, iters)