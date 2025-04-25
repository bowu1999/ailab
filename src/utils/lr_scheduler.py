from torch.optim import lr_scheduler

def build_scheduler(cfg, optimizer):
    name = cfg.pop('type')
    if name == 'StepLR': return lr_scheduler.StepLR(optimizer, **cfg)
    # 可扩展其他策略
    raise ValueError(f"Unknown scheduler: {name}")