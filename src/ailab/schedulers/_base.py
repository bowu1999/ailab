import math
import torch

import math
import torch

class WarmupAndCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupAndCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 预热阶段，将last_epoch加1避免学习率为0
            return [base_lr * ((self.last_epoch + 1) / max(1, self.warmup_steps)) for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [base_lr * 0.5 * (1. + math.cos(math.pi * progress)) for base_lr in self.base_lrs]