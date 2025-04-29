# src/ailab/tools/train.py
import os
import torch
import inspect
import importlib
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from ailab.workflow import WorkFlow
from ailab.utils.dist_utils import init_dist, DistributedSampler
from ailab.hooks_extra import TensorboardHook, WandbHook
from ailab.builder import build_dataset, build_model, build_optimizer, build_loss


def ailab_train(cfg):
    """基于配置文件的训练接口。"""
    os.makedirs(cfg['work_dir'], exist_ok=True)

    # 初始化分布式环境（返回是否分布式训练）
    is_dist = init_dist(cfg)

    # info
    world_size = dist.get_world_size() if is_dist else 1
    rank = dist.get_rank() if is_dist else 0
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # dataloaders
    loaders = {}
    for phase in ['train', 'val', 'test']:
        if phase in cfg['data']:
            ds = build_dataset(cfg['data'][phase])
            sampler = DistributedSampler(
                ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True) if is_dist else None
            dl_cfg = cfg['data'].get(f'{phase}_dataloader', {}).copy()
            if sampler is not None and 'shuffle' in dl_cfg:
                dl_cfg.pop('shuffle')
            loaders[phase] = DataLoader(ds, sampler=sampler, **dl_cfg)

    # model & optimizer
    model = build_model(cfg['model'])
    model = model.cuda(local_rank)
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = build_optimizer(cfg['optimizer'], model.parameters())

    # criterion
    criterion = build_loss(cfg['loss'])
    if torch.cuda.is_available():
        criterion = criterion.cuda(local_rank)
    if hasattr(criterion, 'weight') and criterion.weight is not None:
        model_for_inspect = model.module if hasattr(model, 'module') else model
        num_classes = None
        for m in reversed(list(model_for_inspect.modules())):
            if hasattr(m, 'out_features'):
                num_classes = m.out_features
                break
        if num_classes is not None and criterion.weight.numel() != num_classes:
            import warnings
            warnings.warn(
                f"Loss weight length {criterion.weight.numel()} != model output classes {num_classes}, removing weight.")
            criterion.register_buffer('weight', None)

    wf = WorkFlow(cfg, model, optimizer, criterion)

    # hooks注册
    hooks = []
    if 'hooks' in cfg:
        if 'metrics' in cfg['hooks'] and cfg['hooks']['metrics'].get('enable', True):
            from ailab.hooks import MetricHook
            hooks.append(MetricHook(cfg['hooks']['metrics']))

        for name, hook_cfg in cfg['hooks'].items():
            if name == 'metrics' or not hook_cfg.get('enable', True):
                continue
            try:
                module = importlib.import_module('ailab.hooks')
                HookClass = getattr(module, hook_cfg['type'])
            except (ImportError, AttributeError):
                module = importlib.import_module('ailab.hooks_extra')
                HookClass = getattr(module, hook_cfg['type'])

            if hook_cfg['type'] == 'DDPHook':
                hooks.append(HookClass())
                continue
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

    # 启动流程
    wf.run(loaders, cfg['workflow'], cfg['total_epochs'])


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train a model with ailab framework')
    parser.add_argument('-c', '--config', required=True, help='path to config file')
    return parser.parse_args()


def main():
    args = parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg_module", args.config)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    try:
        cfg = cfg_module.cfg
    except AttributeError:
        raise RuntimeError(f"Config file {args.config} must define a top-level `cfg` dict")

    ailab_train(cfg)


if __name__ == '__main__':
    main()