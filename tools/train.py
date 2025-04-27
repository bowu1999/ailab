import os
import sys

# 动态添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)


import yaml
import inspect
import argparse
from torch.utils.data import DataLoader

from src.workflow import WorkFlow
from src.utils.dist_utils import init_dist, DistSampler
from src.hooks_extra import TensorboardHook, WandbHook
from src.builder import build_dataset, build_model, build_optimizer


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
            dl_cfg = cfg['data'][f'{phase}_dataloader'].copy()
            if sampler is not None and 'shuffle' in dl_cfg:
                dl_cfg.pop('shuffle')
            loaders[phase] = DataLoader(ds, sampler=sampler, **dl_cfg)

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
    for name, hook_cfg in cfg['hooks'].items():
        if not hook_cfg.get('enable', True):
            continue
        # 动态加载 HookClass
        module = __import__('src.hooks', fromlist=[hook_cfg['type']])
        if hasattr(module, hook_cfg['type']):
            HookClass = getattr(module, hook_cfg['type'])
        else:
            module_extra = __import__('src.hooks_extra', fromlist=[hook_cfg['type']])
            HookClass = getattr(module_extra, hook_cfg['type'])
        if HookClass is TensorboardHook:
            hook_cfg['work_dir'] = cfg['work_dir']
            hooks.append(HookClass(hook_cfg))
        # 根据 __init__ 参数决定传参
        sig = inspect.signature(HookClass.__init__)
        param_names = [p for p in sig.parameters.keys() if p != 'self']

        if 'optimizer' in param_names:
            hooks.append(HookClass(hook_cfg, optimizer))
        elif 'cfg' in param_names or 'hook_cfg' in param_names:
            hooks.append(HookClass(hook_cfg))
        else:
            hooks.append(HookClass())

    wf.register_hooks(hooks)
    wf.run(loaders, cfg['workflow'], cfg['total_epochs'])


if __name__ == '__main__':
    main()