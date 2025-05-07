import os
import torch
import torch.distributed as dist

from ailab.builder import build_metrics, build_scheduler
from .utils.logging import get_logger
from .utils.checkpoint import Checkpointer
from .utils.call_fn import call_fn


class Hook:
    def before_run(self, wf): pass
    def after_run(self, wf): pass
    def before_train_epoch(self, wf): pass
    def after_train_epoch(self, wf): pass
    def before_train_iter(self, wf): pass
    def after_train_iter(self, wf): pass
    def before_val_epoch(self, wf): pass
    def after_val_epoch(self, wf): pass
    def before_test_epoch(self, wf): pass
    def after_test_epoch(self, wf): pass


def get_by_path(obj, path, default=None):
    """
    支持两种 path 形式：
      - "optimizer.param_groups.0.lr"
      - "metrics['top1']"
    """
    try:
        cur = obj
        # 先把 ["key"] 之类统一成 .key 格式
        path = path.replace("['", ".").replace("']", "")
        parts = path.split('.')
        for p in parts:
            if p.isdigit():
                cur = cur[int(p)]
            else:
                # dict 访问
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    cur = getattr(cur, p)
        return cur
    except Exception:
        return default


def _expand_path(path: str, cfg: dict) -> str:
    """
    1) 展开 shell 环境变量，例如 ${HOME}、$USER 等
    2) 展开 ~
    3) 简单替换 cfg 里的变量，比如 ${work_dir}
    """
    # 先做 shell env 展开和 ~
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)

    # 再做 cfg 里的简单占位替换
    # 支持 "${work_dir}" 或 "{work_dir}" 两种形式
    for k, v in cfg.items():
        placeholder1 = "${%s}" % k
        placeholder2 = "{%s}" % k
        if placeholder1 in path:
            path = path.replace(placeholder1, str(v))
        if placeholder2 in path:
            path = path.replace(placeholder2, str(v))
    return path


class LoggerHook(Hook):
    """
    - interval: 打印间隔
    - log_items: dict, key->(str path 或 callable(wf)->value)
    """
    def __init__(self, cfg):
        self.interval = cfg['interval']
        self.log_items = cfg.get('log_items', {})
        log_file = None
        if cfg.get('log_dir'):
            os.makedirs(cfg['log_dir'], exist_ok=True)
            log_file = os.path.join(cfg['log_dir'], 'train.log')
        self.logger = get_logger(log_file=log_file)

    def before_run(self, wf):
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        self.logger.info("===== Starting run =====")

    def after_run(self, wf):
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        self.logger.info("===== Run completed =====")

    def before_train_epoch(self, wf):
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        self.logger.info(f"--- Starting train epoch {wf.epoch} ---")

    def after_train_epoch(self, wf):
        # 仅 rank0 打印
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        # 打印该 epoch 的汇总信息
        msgs = self._gather_log_info(wf, 'train')
        self.logger.info("  ".join(msgs))

    def before_val_epoch(self, wf):
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        wf.val_iter = 0
        self.logger.info(f"--- Starting validation epoch {wf.epoch} ---")

    def after_val_epoch(self, wf):
        # 仅 rank0 打印
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        # 打印该验证 epoch 的汇总信息
        msgs = self._gather_log_info(wf, 'val')
        self.logger.info("  ".join(msgs))

    def before_test_epoch(self, wf):
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        wf.val_iter = 0
        self.logger.info(f"--- Starting test epoch {wf.epoch} ---")

    def after_test_epoch(self, wf):
        # 仅 rank0 打印
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        # 打印该测试 epoch 的汇总信息
        msgs = self._gather_log_info(wf, 'test')
        self.logger.info("  ".join(msgs))

    def _should_log(self, wf, phase):
        # 仅 rank0 打印
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return False

        if phase == 'train':
            return (wf.iter + 1) % self.interval == 0
        else:
            return (wf.val_iter + 1) % self.interval == 0

    def _gather_log_info(self, wf, phase):
        msgs = [
            f"Phase: {phase.upper()}",
            f"Epoch[{wf.epoch}]"
        ]

        if phase == 'train':
            msgs.append(f"Iter[{wf.iter + 1}/{wf.max_iter}]")
            msgs.append(f"Loss: {wf.last_loss:.8f}")
        else:
            msgs.append(f"ValIter[{wf.val_iter + 1}]")
            msgs.append(f"Loss: {wf.last_loss:.8f}")

        # 加上用户自定义 log_items
        for name, handler in self.log_items.items():
            if callable(handler):
                val = handler(wf)
            else:
                val = get_by_path(wf, handler, default=None)
            if isinstance(val, float):
                msgs.append(f"{name}: {val:.8f}")
            else:
                msgs.append(f"{name}: {val}")
        return msgs

    def after_train_iter(self, wf):
        if not self._should_log(wf, 'train'):
            return
        msgs = self._gather_log_info(wf, 'train')
        self.logger.info("  ".join(msgs))

    def after_val_iter(self, wf):
        if not hasattr(wf, 'val_iter'):
            wf.val_iter = 0
        if not self._should_log(wf, 'val'):
            wf.val_iter += 1
            return
        msgs = self._gather_log_info(wf, 'val')
        self.logger.info("  ".join(msgs))
        wf.val_iter += 1

    def after_test_iter(self, wf):
        if not hasattr(wf, 'val_iter'):
            wf.val_iter = 0
        if not self._should_log(wf, 'test'):
            wf.val_iter += 1
            return
        msgs = self._gather_log_info(wf, 'test')
        self.logger.info("  ".join(msgs))
        wf.val_iter += 1


class CheckpointHook(Hook):
    def __init__(self, cfg):
        self.interval = cfg['interval']
        self.save_dir = cfg.get('save_dir', None)
        self.checkpointer = None

    def after_train_epoch(self, wf):
        if self.checkpointer is None:
            save_dir = self.save_dir if self.save_dir is not None else wf.work_dir
            self.checkpointer = Checkpointer(save_dir)
        if (wf.epoch + 1) % self.interval == 0:
            self.checkpointer.save(wf.model, wf.optimizer, wf.epoch)


class ResumeHook(Hook):
    def __init__(self, cfg): 
        self.resume_path = cfg.get('resume_from', None)

    def before_run(self, wf):
        if self.resume_path:
            epoch = Checkpointer(wf.work_dir).resume(wf.model, wf.optimizer, self.resume_path)
            wf.epoch = epoch


class LrSchedulerHook(Hook):
    def __init__(self, cfg, optimizer, total_steps = None):
        # cfg 是 hooks.lr_scheduler 部分，
        # 里面须包含一个子字段 scheduler
        scheduler_cfg = cfg.get('scheduler', {})
        
        if not scheduler_cfg or 'type' not in scheduler_cfg:
            raise ValueError("`hooks.lr_scheduler.scheduler.type` must be defined")
        self.scheduler = build_scheduler(scheduler_cfg, optimizer, total_steps)

    def after_train_epoch(self, wf):
        self.scheduler.step()


class DDPHook(Hook):
    def before_run(self, wf):
        from .utils.dist_utils import init_dist
        wf.is_dist = init_dist(wf.cfg)
        if wf.is_dist:
            wf.model = torch.nn.parallel.DistributedDataParallel(
                wf.model.cuda(), device_ids=[int(os.environ.get('LOCAL_RANK', 0))]
            )


class AMPHook(Hook):
    def __init__(self, cfg):
        from torch.amp import GradScaler, autocast
        self.scaler = GradScaler('cuda', enabled=cfg.get('enable', False))
        self.autocast = autocast
        # 兼容两种字段名称，并给出默认值 False
        self.enabled = cfg.get('enable', cfg.get('enabled', False))

    def before_train_iter(self, wf): 
        pass

    def after_train_iter(self, wf): 
        pass

    def wrap_forward(self, func, *args, **kwargs):
        if self.enabled:
            with self.autocast(): return func(*args, **kwargs)

        return func(*args, **kwargs)
    

class MetricHook(Hook):
    def __init__(self, cfg):
        super().__init__()
        # 读取所有配置，确保每个都含 phases、mapping
        self.metric_cfg = {}
        for name, spec in cfg.items():
            if name == 'type': 
                continue
            s = spec.copy()
            s.setdefault('phases', ['val', 'test'])  # 默认只在 val/test
            self.metric_cfg[name] = s

    def before_run(self, wf):
        # 构建所有 metric 实例（去掉 mapping/phases）
        spec_for_build = {
            n: {k: v for k, v in s.items() if k not in ('mapping', 'phases')}
            for n, s in self.metric_cfg.items()
        }
        wf.metrics = build_metrics(spec_for_build)
        wf.metric_results = {}
        # 初始 meters 也挂所有（会在各阶段前重置）
        wf.meters = {}
        for name, m in wf.metrics.items():
            m._mapping = self.metric_cfg[name]['mapping']
            m._phases  = self.metric_cfg[name]['phases']
            wf.meters[name] = m
        device = next(wf.model.parameters()).device
        # 把所有 metric 的内部状态移动到同一个设备
        for m in wf.metrics.values():
            # 只有 torchmetrics.Metric 或继承自 nn.Module 的才调用 .to()
            if hasattr(m, 'to') and callable(m.to):
                try:
                    m.to(device)
                except Exception:
                    # 万一其它指标有 to() 但不支持也不阻塞
                    pass


    def _reset(self, wf, phase):
        # 重置并仅保留本阶段指标
        new_meters = {}
        for name, m in wf.metrics.items():
            if phase in m._phases:
                m.reset()
                new_meters[name] = m
        wf.meters = new_meters

    def _update(self, wf, phase):
        data = wf.last_data
        for name, m in wf.meters.items():  # 只对当前 meters 更新
            call_fn(m.update, data, mapping=m._mapping)

    def _compute(self, wf, phase):
        for name, m in wf.meters.items():
            wf.metric_results[f"{phase}_{name}"] = m.compute()

    # 在不同阶段绑定 reset/update/compute
    def before_train_epoch(self, wf): self._reset(wf, 'train')
    def after_train_iter(self,   wf): self._update(wf, 'train')
    def after_train_epoch(self,   wf): self._compute(wf, 'train')

    def before_val_epoch(self,   wf): self._reset(wf, 'val')
    def after_val_iter(self,     wf): self._update(wf, 'val')
    def after_val_epoch(self,    wf): self._compute(wf, 'val')

    def before_test_epoch(self,  wf): self._reset(wf, 'test')
    def after_test_iter(self,    wf): self._update(wf, 'test')
    def after_test_epoch(self,   wf): self._compute(wf, 'test')