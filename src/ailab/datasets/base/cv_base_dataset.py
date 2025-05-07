from PIL import Image
from typing import Callable, Any

from ..utils import load_dicts_from_jsonlines

from ._transform import std_transform
from ._base import (
    FileAnnotationDataset,
    ClassificationTask
)



class ClassificationImageDataset(FileAnnotationDataset, ClassificationTask):
    """
    JSONL 格式标注 + 图像分类任务

    参数：
      annotation_file (str): JSONL 标注文件路径，每行是一个 dict
      x_key (str): 样本中图像路径字段名
      y_key (str): 样本中标签字段名
      transform (Callable): PIL 图像预处理函数
    返回：
      {'input': Tensor, 'label': int}
    """
    def __init__(
        self,
        annotation_file: str,
        x_key: str,
        y_key: str,
        transform: Callable = std_transform,
    ):
        # 初始化父类，读取 JSONL 到列表 of dict
        FileAnnotationDataset.__init__(self, annotation_file, load_dicts_from_jsonlines)
        self.x_key = x_key
        self.y_key = y_key
        self.transform = transform

    def load_input(self, raw: dict) -> Any:
        path = raw[self.x_key]
        img = Image.open(path)
        return self.transform(img)

    def load_label(self, raw: dict) -> int:
        return int(raw[self.y_key])