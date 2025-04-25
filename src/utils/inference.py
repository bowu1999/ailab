import torch
from ..builder import build_model
from .checkpoint import Checkpointer

def inference(cfg, ckpt_path, input_data):
    model = build_model(cfg['model'])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    # 加载权重
    Checkpointer(cfg['runner']['work_dir']).resume(model, optimizer, ckpt_path)
    model.eval()
    with torch.no_grad():
        return model(**input_data)