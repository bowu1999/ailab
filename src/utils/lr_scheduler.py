# src/utils/lr_scheduler.py
import importlib
from torch.optim import lr_scheduler as _torch_schedulers

def build_scheduler(cfg: dict, optimizer):
    """
    动态构建学习率调度器。
    - 优先在 torch.optim.lr_scheduler 下查找同名类；
    - 否则当作“模块路径.类名”去 import，
      支持用户自定义调度器。
    """
    cfg = cfg.copy()
    scheduler_type = cfg.pop('type')

    # 1) 尝试 PyTorch 自带
    if hasattr(_torch_schedulers, scheduler_type):
        cls = getattr(_torch_schedulers, scheduler_type)
    else:
        # 2) 动态 import 自定义类
        module_name, class_name = scheduler_type.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

    # 剩余 cfg 全部作为参数传给构造函数
    return cls(optimizer, **cfg)
