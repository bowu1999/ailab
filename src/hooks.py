import os
import torch
from .utils.logging import get_logger
from .utils.checkpoint import Checkpointer
from .utils.lr_scheduler import build_scheduler

class Hook:
    def before_run(self, wf): pass
    def after_run(self, wf): pass
    def before_train_epoch(self, wf): pass
    def after_train_epoch(self, wf): pass
    def before_train_iter(self, wf): pass
    def after_train_iter(self, wf): pass
    def before_val_epoch(self, wf): pass
    def after_val_epoch(self, wf): pass
    def before_test_epoch(self, wf): pass
    def after_test_epoch(self, wf): pass

class LoggerHook(Hook):
    def __init__(self, cfg):
        self.interval = cfg['interval']
        log_file = os.path.join(cfg.get('log_dir', ''), 'train.log') if cfg.get('log_dir') else None
        self.logger = get_logger(log_file=log_file)
    def after_train_iter(self, wf):
        if wf.iter % self.interval == 0:
            self.logger.info(f"Epoch[{wf.epoch}] Iter[{wf.iter}] Loss: {wf.last_loss}")

class CheckpointHook(Hook):
    def __init__(self, cfg):
        self.interval = cfg['interval']
        self.checkpointer = Checkpointer(cfg.get('save_dir', wf.work_dir))
    def after_train_epoch(self, wf):
        if (wf.epoch + 1) % self.interval == 0:
            self.checkpointer.save(wf.model, wf.optimizer, wf.epoch)

class ResumeHook(Hook):
    def __init__(self, cfg): self.resume_path = cfg.get('resume_from')
    def before_run(self, wf):
        if self.resume_path:
            epoch = Checkpointer(wf.work_dir).resume(wf.model, wf.optimizer, self.resume_path)
            wf.epoch = epoch

class LrSchedulerHook(Hook):
    def __init__(self, cfg, optimizer):
        self.scheduler = build_scheduler(cfg, optimizer)
    def after_train_epoch(self, wf):
        self.scheduler.step()

class DDPHook(Hook):
    def before_run(self, wf):
        from .utils.dist_utils import init_dist
        wf.is_dist = init_dist(wf.cfg)
        if wf.is_dist:
            wf.model = torch.nn.parallel.DistributedDataParallel(
                wf.model.cuda(), device_ids=[int(os.environ.get('LOCAL_RANK', 0))]
            )

class AMPHook(Hook):
    def __init__(self, cfg):
        from torch.cuda.amp import GradScaler, autocast
        self.scaler = GradScaler()
        self.autocast = autocast
        self.enabled = cfg['enabled']
    def before_train_iter(self, wf): pass
    def after_train_iter(self, wf): pass
    def wrap_forward(self, func, *args, **kwargs):
        if self.enabled:
            with self.autocast(): return func(*args, **kwargs)
        return func(*args, **kwargs)