from .registry import DATASETS, MODELS, OPTIMIZERS


# Builder utilities
def build_dataset(cfg):
    return DATASETS.build(cfg)

def build_model(cfg):
    return MODELS.build(cfg)

def build_optimizer(cfg, parameters):
    opt = OPTIMIZERS.build(cfg, default_args={'params': parameters})
    return opt