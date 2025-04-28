import torch
from torch import distributed as dist

from .registry import DATASETS, MODELS, OPTIMIZERS, METRICS, LOSSES


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
        print(f"Building dataset: {model.__class__.__name__}")
    return model


def build_optimizer(cfg, parameters):
    opt = OPTIMIZERS.build(cfg, default_args={'params': parameters})
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Building dataset: {opt.__class__.__name__}")
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
    return metric_dict


def build_loss(cfg):
    """Build loss function from config dict, converting list weights to tensor."""
    cfg = cfg.copy()
    # Convert weight list to torch.Tensor if provided
    if 'weight' in cfg and isinstance(cfg['weight'], list):
        cfg['weight'] = torch.tensor(cfg['weight'], dtype=torch.float)
    # Other loss kwargs remain as-is
    return LOSSES.build(cfg)