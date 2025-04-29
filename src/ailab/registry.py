# src/ailab/registry.py
import os
import sys
import pkgutil
import inspect
import importlib
import torch.nn as nn
from torch import distributed as dist
from torch.nn.modules.loss import _Loss
from torch.optim import lr_scheduler as _torch_schedulers

# ---------------------------------------------------
# Allow running without installation by adding src to PYTHONPATH
# ---------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# ---------------------------------------------------

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
            # raise KeyError(f"'{mod_name}' is already registered")
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                print(f"'{mod_name}' is already registered, raw '{mod_name}' will be cover")
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
                raise KeyError(f"'{obj_type}' is not registered")
        else:
            obj_cls = obj_type
        if default_args:
            for n, v in default_args.items():
                args.setdefault(n, v)
        return obj_cls(**args)

# Create global registries
DATASETS      = Registry()
MODELS        = Registry()
OPTIMIZERS    = Registry()
LR_SCHEDULERS = Registry()
METRICS       = Registry()
LOSSES        = Registry()


def auto_import_and_register_packages(package_path, package_name, registry):
    """
    递归导入目录下所有模块并自动注册所有类和函数
    """
    if not os.path.isdir(package_path):
        return

    for finder, name, ispkg in pkgutil.iter_modules([package_path]):
        full_module_name = f"{package_name}.{name}"
        module = importlib.import_module(full_module_name)

        # 递归处理子包
        if ispkg:
            sub_pkg_path = os.path.join(package_path, name)
            sub_pkg_name = full_module_name
            auto_import_and_register_packages(sub_pkg_path, sub_pkg_name, registry)

        # 遍历模块中所有类、函数
        for obj_name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) or inspect.isfunction(obj)) and obj.__module__ == full_module_name:
                try:
                    registry.register_module(obj)
                except Exception as e:
                    print(f"Warning: {obj_name} cannot be registered: {e}")


# -----------------------
# 自动注册 torchvision.datasets 中的所有 Dataset 子类
# -----------------------
try:
    import torchvision.datasets as tv_datasets
    from torch.utils.data import Dataset as _TorchDataset
    for name, obj in inspect.getmembers(tv_datasets):
        if inspect.isclass(obj) and issubclass(obj, _TorchDataset):
            DATASETS.register_module(obj, name=name)
except ImportError:
    pass

# -----------------------
# 自动递归导入&注册 datasets
# -----------------------
_datasets_pkg = os.path.abspath(os.path.join(current_dir, 'datasets'))
if os.path.isdir(_datasets_pkg):
    auto_import_and_register_packages(_datasets_pkg, 'ailab.datasets', DATASETS)


# -----------------------
# 自动注册 torchvision.models 中的所有模型构建函数/类
# -----------------------
try:
    import torchvision.models as tv_models
    for name, obj in inspect.getmembers(tv_models):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            MODELS.register_module(obj, name=name)
except ImportError:
    pass

# -----------------------
# 自动递归导入&注册 models
# -----------------------
_models_pkg = os.path.abspath(os.path.join(current_dir, 'models'))
if os.path.isdir(_models_pkg):
    auto_import_and_register_packages(_models_pkg, 'ailab.models', MODELS)


# -----------------------
# 注册 PyTorch 内置 optimizers
# -----------------------
import torch.optim as optim
for name in dir(optim):
    opt = getattr(optim, name)
    if isinstance(opt, type) and issubclass(opt, optim.Optimizer) and opt is not optim.Optimizer:
        try:
            OPTIMIZERS.register_module(opt, name=name)
        except Exception:
            pass

# -----------------------
# 自动递归导入&注册 用户自定义 optimizers
# -----------------------
_optim_pkg = os.path.abspath(os.path.join(current_dir, 'optimizers'))
if os.path.isdir(_optim_pkg):
    auto_import_and_register_packages(_optim_pkg, 'ailab.optimizers', OPTIMIZERS)


# -----------------------
# 注册 PyTorch.optim.lr_scheduler 内置全部调度器
# -----------------------
for name, obj in inspect.getmembers(_torch_schedulers):
    if inspect.isclass(obj):
        try:
            LR_SCHEDULERS.register_module(obj, name=name)
        except Exception:
            pass

# -----------------------
# 自动递归导入&注册 lr_schedulers
# -----------------------
_schedulers_pkg = os.path.abspath(os.path.join(current_dir, 'schedulers'))
if os.path.isdir(_schedulers_pkg):
    auto_import_and_register_packages(_schedulers_pkg, 'ailab.schedulers', LR_SCHEDULERS)


# -----------------------
# 自动注册 torchmetrics 库中的所有 Metric 类
# -----------------------
try:
    import torchmetrics
    for module_info in pkgutil.walk_packages(torchmetrics.__path__, torchmetrics.__name__ + '.'):
        module = importlib.import_module(module_info.name)
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                try:
                    METRICS.register_module(obj, name=name)
                except Exception:
                    pass
except ImportError:
    pass

# -----------------------
# 自动递归导入&注册 metrics
# -----------------------
_metrics_pkg = os.path.abspath(os.path.join(current_dir, 'metrics'))
if os.path.isdir(_metrics_pkg):
    auto_import_and_register_packages(_metrics_pkg, 'ailab.metrics', METRICS)


# -----------------------
# 注册 PyTorch 内置 losses
# -----------------------
for name, obj in inspect.getmembers(nn):
    if inspect.isclass(obj) and issubclass(obj, _Loss) and obj is not _Loss:
        try:
            LOSSES.register_module(obj, name=name)
        except Exception:
            pass

# -----------------------
# 自动递归导入&注册 用户自定义 losses
# -----------------------
_losses_pkg = os.path.abspath(os.path.join(current_dir, 'losses'))
if os.path.isdir(_losses_pkg):
    auto_import_and_register_packages(_losses_pkg, 'ailab.losses', LOSSES)
