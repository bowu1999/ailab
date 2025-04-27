import os
import sys

# 动态添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)


import torch
import inspect
import argparse
import importlib.util
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.workflow import WorkFlow
from src.utils.dist_utils import init_dist, DistSampler
from src.hooks_extra import TensorboardHook, WandbHook
from src.builder import build_dataset, build_model, build_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with ailab framework')
    parser.add_argument('-c', '--config', help='path to config file')
    return parser.parse_args()


def _is_noarg_init(hook_cls):
    sig = inspect.signature(hook_cls.__init__)
    params = [p for p in sig.parameters.keys() if p != 'self']
    return len(params) == 0


def main():
    args = parse_args()
    # ----- load Python config module -----
    spec = importlib.util.spec_from_file_location("cfg_module", args.config)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    # your config file must define a dict named `cfg`
    try:
        cfg = cfg_module.cfg
    except AttributeError:
        raise RuntimeError(f"Config file {args.config} must define a top-level `cfg` dict")
    os.makedirs(cfg['work_dir'], exist_ok=True)

    # init distributed
    is_dist = init_dist(cfg)

    # dataloaders
    loaders = {}
    for phase in ['train', 'val', 'test']:
        if phase in cfg['data']:
            ds = build_dataset(cfg['data'][phase])
            sampler = DistSampler(ds) if is_dist and phase == 'train' else None
            dl_cfg = cfg['data'][f'{phase}_dataloader'].copy()
            if sampler is not None and 'shuffle' in dl_cfg:
                dl_cfg.pop('shuffle')
            loaders[phase] = DataLoader(ds, sampler=sampler, **dl_cfg)
    # print(f"train samples: {len(loaders['train'].dataset)}")

    # model & optimizer
    model = build_model(cfg['model'])
    optimizer = build_optimizer(cfg['optimizer'], model.parameters())

    # workflow
    from torch import nn
    loss_cfg = cfg.get('loss', {})
    criterion = getattr(nn, loss_cfg.get('type', 'CrossEntropyLoss'))(**loss_cfg.get('kwargs', {}))
    # 传入 criterion
    wf = WorkFlow(cfg, model, optimizer, criterion)

    # hooks registration
    hooks = []
    # 先把 MetricHook 放进来
    if 'metrics' in cfg['hooks'] and cfg['hooks']['metrics'].get('enable', True):
        from src.hooks import MetricHook
        hooks.append(MetricHook(cfg['hooks']['metrics']))
    # 然后再按常规流程注册其它 Hook
    for name, hook_cfg in cfg['hooks'].items():
        if name == 'metrics' or not hook_cfg.get('enable', True):
            continue
        module = __import__('src.hooks', fromlist=[hook_cfg['type']])
        HookClass = getattr(module, hook_cfg['type'], None) \
                    or getattr(__import__('src.hooks_extra', fromlist=[hook_cfg['type']]), hook_cfg['type'])
        if hook_cfg['type'] == 'DDPHook':
            hooks.append(HookClass())
            continue
        # Tensorboard 特殊传 work_dir
        if HookClass is TensorboardHook:
            hook_cfg['work_dir'] = cfg['work_dir']
            hooks.append(HookClass(hook_cfg))
            continue
        sig = inspect.signature(HookClass.__init__)
        params = [p for p in sig.parameters if p != 'self']
        if 'optimizer' in params:
            hooks.append(HookClass(hook_cfg, optimizer))
        else:
            hooks.append(HookClass(hook_cfg))

    wf.register_hooks(hooks)
    wf.run(loaders, cfg['workflow'], cfg['total_epochs'])


if __name__ == '__main__':
    main()