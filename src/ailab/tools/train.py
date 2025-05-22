# src/ailab/tools/train.py
import os
import torch
import inspect
import importlib
from math import ceil
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from ailab.workflow import WorkFlow
from ailab.utils.dist_utils import init_dist, DistributedSampler
from ailab.hooks_extra import TensorboardHook, WandbHook
from ailab.builder import build_dataset, build_model, build_optimizer, build_loss
from ailab.utils import call_fn, OutputWrapper


def ailab_train(cfg):
    """基于配置文件的训练接口。"""
    # 启用异常检测，找到 in-place 操作的源头
    torch.autograd.set_detect_anomaly(True)
    os.makedirs(cfg['work_dir'], exist_ok = True)

    # 初始化分布式环境（返回是否分布式训练）
    is_dist = init_dist(cfg)

    # 训练配置
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if is_dist else 0
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # 数据集及数据加载器
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

    # 模型和优化器
    model = build_model(cfg['model']).cuda(local_rank)
    # 先 wrap 输出，设置 signature
    wrapped = OutputWrapper(model, output_keys=cfg['model'].get('output_keys', ['output']))
    # 然后再作 DDP
    if is_dist:
        model = DDP(
            wrapped,
            device_ids = [local_rank],
            output_device = local_rank,
            # find_unused_parameters = True,
            static_graph = True
        )
    else:
        model = wrapped
    optimizer = build_optimizer(cfg['optimizer'], model.parameters())

    # 损失函数
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
                f"Loss weight length {criterion.weight.numel()} "\
                    "!= model output classes {num_classes}, removing weight.")
            criterion.register_buffer('weight', None)

    # 实例化 WorkFlow
    wf = WorkFlow(cfg, model, optimizer, criterion)

    # 计算总共的 steps（total_steps）
    dataset_size = len(loaders['train'].dataset)
    batch_size = cfg.get("data").get("train_dataloader").get("batch_size")
    num_epochs = cfg.get("total_epochs")
    total_steps = num_epochs * ceil(dataset_size / batch_size / world_size)

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
                hooks.append(HookClass(hook_cfg, optimizer, total_steps))
            else:
                hooks.append(HookClass(hook_cfg))
    wf.register_hooks(hooks)

    # 启动流程
    wf.run(loaders, cfg['workflow'], cfg['total_epochs'])
    if is_dist:
        dist.destroy_process_group()


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
