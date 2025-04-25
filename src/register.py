"""
OpenMMLab-Style Framework Skeleton

This module provides a basic OpenMMLab-like framework for:
  - Config loading
  - Registries and builder utilities
  - Dataset, model, optimizer building
  - Runner with training, testing, deployment

Note: This skeleton retains core design patterns from OpenMMLab (registry, config-driven instantiation, runner, hooks).
"""
import os
import yaml
import importlib
from collections import defaultdict


def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


class Registry:
    """A registry to map strings to modules/classes."""
    def __init__(self):
        self._module_dict = {}

    def register_module(self, module=None, name=None):
        # Allows use as decorator or direct call
        if module is None:
            return lambda x: self.register_module(x, name=name)
        mod_name = name or module.__name__
        if mod_name in self._module_dict:
            raise KeyError(f'{mod_name} is already registered')
        self._module_dict[mod_name] = module
        return module

    def get(self, name):
        return self._module_dict.get(name)

    def build(self, cfg, default_args=None):
        if not isinstance(cfg, dict) or 'type' not in cfg:
            raise TypeError('cfg must be a dict and contain the key "type"')
        args = cfg.copy()
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
            if obj_cls is None:
                raise KeyError(f'{obj_type} is not registered')
        else:
            obj_cls = obj_type
        if default_args:
            for name, value in default_args.items():
                args.setdefault(name, value)
        return obj_cls(**args)


# Create global registries
DATASETS = Registry()
MODELS = Registry()
OPTIMIZERS = Registry()


# 自动导入 models 包以注册模型
pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
for _, module_name, _ in pkgutil.iter_modules([pkg_dir]):
    importlib.import_module(f"models.{module_name}")