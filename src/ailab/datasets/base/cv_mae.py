import torch
import numpy as np
from PIL import Image
from typing import Callable, Any

from ..utils import read_image_paths_from_file

from ._transform import std_transform
from ._base import FileAnnotationDataset


class MaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.num_patches = input_size[0] * input_size[1]
        self.mask_ratio = mask_ratio

    def __call__(self):
        mask = np.hstack([
            np.zeros(int((1 - self.mask_ratio) * self.num_patches)),
            np.ones(int(self.mask_ratio * self.num_patches))
        ])
        np.random.shuffle(mask)
        return torch.from_numpy(mask.astype(bool))


class MAEAnnFileDataset(FileAnnotationDataset):
    """
    JSONL 格式标注 + MAE 预训练任务

    参数：
      annotation_file (str): 每行是图像路径的文件
      img_size (int), patch_size (int), mask_ratio (float)：MaskingGenerator 配置
      transform (Callable)：PIL 图像预处理函数
    返回：
      {'image': Tensor, 'mask': Tensor, 'target': Tensor}
    """
    def __init__(
        self,
        annotation_file: str,
        img_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        transform: Callable = std_transform,
    ):
        # 每行是一个图像路径字符串
        FileAnnotationDataset.__init__(self, annotation_file, read_image_paths_from_file)
        self.transform = transform
        self.mask_gen = MaskingGenerator(
            (img_size // patch_size, img_size // patch_size),
            mask_ratio
        )

    def _get_sample(self, raw: Any) -> dict:
        # raw 已经是 image_path 字符串
        img = Image.open(raw)
        img = self.transform(img)
        mask = self.mask_gen()  # 二值 mask
        return {
            'image': img,
            'mask': mask.bool(),
            'target': img
        }