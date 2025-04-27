import os
import torch
import torch.distributed as dist

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
        # 只让主进程打印
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        if wf.iter % self.interval == 0:
            self.logger.info(f"Epoch[{wf.epoch}] Iter[{wf.iter}] Loss: {wf.last_loss}")


class CheckpointHook(Hook):
    def __init__(self, cfg):
        self.interval = cfg['interval']
        self.save_dir = cfg.get('save_dir', None)
        self.checkpointer = None

    def after_train_epoch(self, wf):
        if self.checkpointer is None:
            save_dir = self.save_dir if self.save_dir is not None else wf.work_dir
            self.checkpointer = Checkpointer(save_dir)
        if (wf.epoch + 1) % self.interval == 0:
            self.checkpointer.save(wf.model, wf.optimizer, wf.epoch)


class ResumeHook(Hook):
    def __init__(self, cfg): 
        self.resume_path = cfg.get('resume_from')

    def before_run(self, wf):
        if self.resume_path:
            epoch = Checkpointer(wf.work_dir).resume(wf.model, wf.optimizer, self.resume_path)
            wf.epoch = epoch


class LrSchedulerHook(Hook):
    def __init__(self, cfg, optimizer):
        # cfg 是 hooks.lr_scheduler 部分，
        # 里面须包含一个子字段 scheduler
        scheduler_cfg = cfg.get('scheduler', {})
        if not scheduler_cfg or 'type' not in scheduler_cfg:
            raise ValueError("`hooks.lr_scheduler.scheduler.type` must be defined")
        self.scheduler = build_scheduler(scheduler_cfg, optimizer)

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
        from torch.amp import GradScaler, autocast
        self.scaler = GradScaler('cuda', enabled=cfg.get('enable', False))
        self.autocast = autocast
        # 兼容两种字段名称，并给出默认值 False
        self.enabled = cfg.get('enable', cfg.get('enabled', False))

    def before_train_iter(self, wf): 
        pass

    def after_train_iter(self, wf): 
        pass

    def wrap_forward(self, func, *args, **kwargs):
        if self.enabled:
            with self.autocast(): return func(*args, **kwargs)

        return func(*args, **kwargs)