import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def init_dist(cfg):
    if dist.is_available() and not dist.is_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=cfg['dist']['backend'],
            init_method=cfg['dist']['init_method'],
            world_size=int(os.environ.get('WORLD_SIZE', cfg['dist']['world_size'])),
            rank=int(os.environ.get('RANK', 0))
        )
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        return True
    return False


class DistSampler(DistributedSampler):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)