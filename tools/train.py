import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from src.utils.dist_utils import init_dist, DistSampler
from src.builder import build_dataset, build_model, build_optimizer
from src.workflow import WorkFlow
from src.hooks import (LoggerHook, CheckpointHook,
                       ResumeHook, LrSchedulerHook,
                       DDPHook, AMPHook)
from src.hooks_extra import TensorboardHook, WandbHook


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with ailab framework')
    parser.add_argument('-c', '--config', help='path to config file')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    os.makedirs(cfg['work_dir'], exist_ok=True)

    # init distributed
    is_dist = init_dist(cfg)
    # dataloaders
    loaders = {}
    for phase in ['train', 'val', 'test']:
        if phase in cfg['data']:
            ds = build_dataset(cfg['data'][phase])
            sampler = DistSampler(ds) if is_dist and phase == 'train' else None
            dl_cfg = cfg['data'][f'{phase}_dataloader']
            loaders[phase] = DataLoader(ds, sampler=sampler, **dl_cfg)

    # model & optimizer
    model = build_model(cfg['model'])
    optimizer = build_optimizer(cfg['optimizer'], model.parameters())

    # workflow
    wf = WorkFlow(cfg, model, optimizer)

    # hooks registration
    hooks = []
    for name, hook_cfg in cfg['hooks'].items():
        if not hook_cfg.get('enable', True):
            continue
        # try import from src.hooks or src.hooks_extra
        if hasattr(__import__('src.hooks', fromlist=[hook_cfg['type']]), hook_cfg['type']):
            HookClass = getattr(__import__('src.hooks', fromlist=[hook_cfg['type']]), hook_cfg['type'])
        else:
            HookClass = getattr(__import__('src.hooks_extra', fromlist=[hook_cfg['type']]), hook_cfg['type'])
        # instantiate
        if 'optimizer' in HookClass.__init__.__code__.co_varnames:
            hooks.append(HookClass(hook_cfg, optimizer))
        else:
            hooks.append(HookClass(hook_cfg))

    wf.register_hooks(hooks)
    wf.run(loaders, cfg['workflow'], cfg['total_epochs'])


if __name__ == '__main__':
    main()