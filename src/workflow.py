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
    def __init__(self, cfg, model, optimizer):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.work_dir = cfg['work_dir']
        os.makedirs(self.work_dir, exist_ok=True)
        self.hooks = []
        self.epoch = 0
        self.iter = 0
        self.last_loss = None

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
        for _ in range(iters):
            self.call_hook('before_train_iter')
            data = next(data_iter)
            out = self.model(**data)
            loss = out.get('loss')
            if loss is None: raise ValueError('Model must return loss for training')
            self.last_loss = loss.item() if hasattr(loss, 'item') else loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iter += 1
            self.call_hook('after_train_iter')

    def _val_epoch(self, loader, iters=None):
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                _ = self.model(**data)

    def _test_epoch(self, loader, iters=None):
        self._val_epoch(loader)