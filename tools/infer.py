import os
import yaml, torch
from src.utils.inference import inference

cfg = yaml.safe_load(open('configs/train.yaml'))
input_data = {'images': torch.randn(1,3,224,224)}
out = inference(cfg, cfg['resume_from'] or os.path.join(cfg['work_dir'], f"epoch_{cfg['total_epochs']}.pth"), input_data)
print(out)