import os
import wandb
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from .hooks import Hook


class TensorboardHook(Hook):
    def __init__(self, cfg):
        work_dir = cfg.get('work_dir', None)
        # if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        #     print(f"work_dir: {work_dir}")
        if work_dir is None:
            raise ValueError("TensorboardHook requires a 'work_dir'—either pass it in or add it to cfg")

        # 输出目录：优先用户在 cfg 中指定的 log_dir，否则用 work_dir/tf_logs
        log_dir = cfg.get('log_dir') or os.path.join(work_dir, 'tf_logs')

        # 保证只有主进程创建文件夹
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                os.makedirs(log_dir, exist_ok=True)
            dist.barrier()
        else:
            os.makedirs(log_dir, exist_ok=True)

        # 仅主进程真正创建 writer
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def after_train_iter(self, wf):
        if self.writer is None:
            return
        self.writer.add_scalar('train/loss', wf.last_loss, wf.iter)

    def after_run(self, wf):
        if self.writer:
            self.writer.close()


class WandbHook(Hook):
    def __init__(self, cfg):
        import wandb
        init_args = cfg.get('init_args', {})
        # 如果配置了 mode，就传给 wandb.init
        if 'mode' in init_args:
            wandb.init(**init_args)
        else:
            # 默认联网模式
            wandb.init(**init_args)