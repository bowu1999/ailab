import torch
from torch import distributed as dist

from .registry import DATASETS, MODELS, OPTIMIZERS, METRICS, LOSSES, LR_SCHEDULERS


# Builder utilities
def build_dataset(cfg):
    ds = DATASETS.build(cfg)
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Building dataset: {ds.__class__.__name__}")
        print(f"Dataset num: {len(ds)}")
    return ds


def build_model(cfg):
    model = MODELS.build(cfg)
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Building model: {model.__class__.__name__}")
    return model


def build_optimizer(cfg, parameters):
    opt = OPTIMIZERS.build(cfg, default_args={'params': parameters})
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Building optimizer: {opt.__class__.__name__}")
    return opt


def build_metrics(cfg: dict):
    """
    cfg 是一个 dict，形如：
      metrics:
        top1:
          type: "Accuracy"
          topk: 1
        top5:
          type: "Accuracy"
          topk: 5

    返回一个 dict{name: metric_obj}
    """
    metric_dict = {}
    for name, spec in cfg.items():
        metric = METRICS.build(spec)
        metric_dict[name] = metric
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Building metric: {metric_dict.__class__.__name__}")
    return metric_dict


def build_loss(cfg):
    """Build loss function from config dict, converting list weights to tensor."""
    cfg = cfg.copy()
    # Convert weight list to torch.Tensor if provided
    if 'weight' in cfg and isinstance(cfg['weight'], list):
        cfg['weight'] = torch.tensor(cfg['weight'], dtype=torch.float)
    # Other loss kwargs remain as-is
    loss_fun = LOSSES.build(cfg)
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Building loss: {loss_fun.__class__.__name__}")
    return loss_fun


def build_scheduler(cfg: dict, optimizer, total_steps = None):
    """Build loss function from config dict, converting list weights to tensor."""
    cfg = cfg.copy()
    lr_scheduler = LR_SCHEDULERS.build(cfg, default_args = {'optimizer': optimizer, 'total_steps': total_steps})
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Building lr_scheduler: {lr_scheduler.__class__.__name__}")
    return lr_scheduler