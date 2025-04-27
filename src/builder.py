from torch import distributed as dist

from .registry import DATASETS, MODELS, OPTIMIZERS


# Builder utilities
def build_dataset(cfg):
    ds = DATASETS.build(cfg)
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Building dataset: {ds.__class__.__name__}")
    return ds

def build_model(cfg):
    return MODELS.build(cfg)

def build_optimizer(cfg, parameters):
    opt = OPTIMIZERS.build(cfg, default_args={'params': parameters})
    return opt