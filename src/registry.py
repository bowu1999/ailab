import os
import pkgutil
import inspect
import importlib


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
            raise KeyError(f"'{mod_name}' is already registered")
        self._module_dict[mod_name] = module
        # print(f"[Registry] Registered {mod_name}")
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
            sub_package_path = os.path.join(package_path, name)
            sub_package_name = f"{package_name}.{name}"
            auto_import_and_register_packages(sub_package_path, sub_package_name, registry)

        # 遍历模块中所有类、函数
        for obj_name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) or inspect.isfunction(obj):
                # 仅注册定义在本模块里的
                if obj.__module__ == full_module_name:
                    try:
                        registry.register_module(obj)
                    except Exception as e:
                        print(f"Warning: {obj_name} cannot be registered: {e}")


# -----------------------
# 自动递归导入&注册 datasets
# -----------------------
_datasets_pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets'))
if os.path.isdir(_datasets_pkg):
    auto_import_and_register_packages(_datasets_pkg, 'src.datasets', DATASETS)

# -----------------------
# 自动递归导入&注册 models
# -----------------------
_models_pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
if os.path.isdir(_models_pkg):
    auto_import_and_register_packages(_models_pkg, 'src.models', MODELS)


import torch.optim as optim


def register_torch_optimizers(registry):
    for name in dir(optim):
        opt = getattr(optim, name)
        if isinstance(opt, type) and issubclass(opt, optim.Optimizer) and opt is not optim.Optimizer:
            try:
                registry.register_module(opt, name=name)
            except Exception as e:
                print(f'Fail to register {name}: {e}')

register_torch_optimizers(OPTIMIZERS)

# -----------------------
# 自动递归导入&注册 optimizers
# -----------------------
_optim_pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), 'optimizers'))
if os.path.isdir(_optim_pkg):
    auto_import_and_register_packages(_optim_pkg, 'src.optimizers', OPTIMIZERS)

# -----------------------
# 自动递归导入&注册 lr_schedulers
# -----------------------
_schedulers_pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), 'schedulers'))
if os.path.isdir(_schedulers_pkg):
    auto_import_and_register_packages(_schedulers_pkg, 'src.schedulers', LR_SCHEDULERS)
