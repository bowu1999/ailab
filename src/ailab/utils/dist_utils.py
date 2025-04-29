# src/ailab/utils/dist_utils.py
import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def init_dist(cfg):
    """
    Initialize distributed training environment.

    Args:
        cfg (dict): Configuration dict. Expected keys:
            - 'launcher': str, e.g., 'pytorch' or 'none';
            - 'dist': dict with keys 'backend', 'init_method'.

    Returns:
        bool: True if distributed has been initialized, False otherwise.
    """
    # Skip distributed if launcher is none
    if cfg.get('launcher', 'pytorch').lower() == 'none':
        return False

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU.")
        return False

    # Initialize only once
    if dist.is_available() and not dist.is_initialized():
        # Set device based on local rank
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        n_gpu = torch.cuda.device_count()
        if local_rank >= n_gpu:
            print(f"WARNING: local_rank {local_rank} >= available GPUs {n_gpu}, defaulting to GPU 0.")
            local_rank = 0
        torch.cuda.set_device(local_rank)

        # Retrieve world size and rank from environment
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', 0))

        # Initialize the default process group
        dist.init_process_group(
            backend=cfg['dist'].get('backend', 'nccl'),
            init_method=cfg['dist'].get('init_method'),
            world_size=world_size,
            rank=rank
        )
        dist.barrier(device_ids=[local_rank])
        return True

    return False


class DistSampler(DistributedSampler):
    """
    A thin wrapper over PyTorch's DistributedSampler.
    """
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
