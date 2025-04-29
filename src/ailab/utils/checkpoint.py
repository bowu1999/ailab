import os
import torch


class Checkpointer:
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def save(self, model, optimizer, epoch):
        path = os.path.join(self.save_dir, f'epoch_{epoch+1}.pth')
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch+1
        }, path)

    def resume(self, model, optimizer, ckpt_path):
        # 设置 map_location 为 'cpu'，以避免加载到不存在的 CUDA 设备上
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        return ckpt['epoch']


def remove_module_prefix(state_dict):
    """
    去掉state_dict中所有key的'module.'前缀。
    
    :param state_dict: 包含模型参数和其值的有序字典。
    :return: 新的有序字典，其中所有的key都没有'module.'前缀。
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]  # 去掉'module.'前缀
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict