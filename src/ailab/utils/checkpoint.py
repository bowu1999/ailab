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
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        return ckpt['epoch']