from .hooks import Hook
from torch.utils.tensorboard import SummaryWriter
import wandb
import os

class TensorboardHook(Hook):
    def __init__(self, cfg):
        log_dir = cfg.get('log_dir', os.getcwd())
        self.tb = SummaryWriter(log_dir)
    def after_train_iter(self, wf):
        self.tb.add_scalar('loss/train', wf.last_loss, wf.iter)

class WandbHook(Hook):
    def __init__(self, cfg):
        wandb.init(project=cfg.get('project', 'ailab'))
    def after_train_iter(self, wf):
        wandb.log({'loss': wf.last_loss, 'epoch': wf.epoch, 'iter': wf.iter})