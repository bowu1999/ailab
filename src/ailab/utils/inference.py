import os
import glob
import torch
from torch.utils.data import DataLoader, Dataset

from ..builder import build_dataset, build_model
from .checkpoint import Checkpointer

class _TempDataset(Dataset):
    """
    用来把一系列文件路径交给已有的 Dataset 实现加载+归一化：
    - dataset_class: e.g. ClassificationImageDataset
    - init_args: 原始 cfg['data'][phase] dict
    - file_list: list of image paths
    """
    def __init__(self, dataset_class, init_args, file_list):
        # 移除原 init_args 里的 annotation_file/data_path，让类接受 file_list
        args = init_args.copy()
        args.pop('annotation_file', None)
        args.pop('data_path', None)
        # 直接传递文件列表
        args['file_list'] = file_list
        self.ds = dataset_class(**args)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # 假设 Dataset 返回 (img_tensor, target) 或 dict
        data = self.ds[idx]
        # 丢弃 target（推理时不需要）
        if isinstance(data, tuple):
            return data[0]
        if isinstance(data, dict) and 'target' in data:
            data.pop('target')
        return data

def inference(cfg: dict, ckpt_path: str, input_path: str):
    # 1. 构建模型 & 加载权重
    model = build_model(cfg['model'])
    model.eval()
    dummy_opt = torch.optim.SGD(model.parameters(), lr=0.)
    Checkpointer(cfg['runner']['work_dir']).resume(model, dummy_opt, ckpt_path)

    def _infer(dataloader):
        results = []
        with torch.no_grad():
            for batch in dataloader:
                # batch 可以是 Tensor 或 dict(inputs,...)
                if isinstance(batch, dict):
                    out = model(**batch)
                else:
                    out = model(batch)
                results.append(out)
        return results

    # 2. 判定输入类型，并生成 file_list 或直接 load .pt
    img_exts = {'.jpg','.jpeg','.png','.bmp','.tiff','.tif'}
    base_data_cfg = cfg['data'].get('test', cfg['data'].get('val'))
    DatasetClass = build_dataset.__self__.get(base_data_cfg['type'])  # or import directly
    if os.path.isdir(input_path):
        # 目录模式：递归查找所有图片
        file_list = []
        for root, _, files in os.walk(input_path):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in img_exts:
                    file_list.append(os.path.join(root, fname))
        tmp_ds = _TempDataset(DatasetClass, base_data_cfg, file_list)
        dl = DataLoader(tmp_ds, batch_size=cfg['data']['val_dataloader']['batch_size'],
                        shuffle=False, **cfg['data']['val_dataloader'].get('extra', {}))
        return _infer(dl)

    ext = os.path.splitext(input_path)[1].lower()
    if ext in img_exts:
        # 单图：file_list 只有一个元素
        tmp_ds = _TempDataset(DatasetClass, base_data_cfg, [input_path])
        dl = DataLoader(tmp_ds, batch_size=1, shuffle=False)
        return _infer(dl)[0]

    if ext in {'.pt', '.pth'}:
        # 直接加载用户输入的 Tensor 或 dict
        data = torch.load(input_path, map_location='cpu')
        with torch.no_grad():
            if isinstance(data, dict):
                return model(**data)
            return model(data)

    raise ValueError(f"Unsupported input path: {input_path}")
