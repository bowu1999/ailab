from pathlib import Path
from typing import Callable
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class AnnotationFileLoadingDataset(Dataset, ABC):
    """
    根据数据和标签文件加载图像和标签，数据和对应的标注保存在一个文件中，文件读取后应该获取一个列表。
    Args:
        annotation_file (string): 包含图像路径和标签的文本文件。
        load_fun (callable): 用于加载图像和标签的函数，接受一个字符串参数（包含图像路径和标签）。
    """
    def __init__(
        self,
        annotation_file: str,
        load_fun: Callable,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"[class {self.__class__.__name__} __init__]: Annotation file not found: {self.annotation_file}"
            )
        else:
            self.samples = load_fun(str(self.annotation_file))

    def __len__(self):
        return self._get_len()

    def __getitem__(self, index):
        sample = self.samples[index]
        return self._get_sample(sample)
    
    def _get_len(self):
        return len(self.samples)

    @abstractmethod
    def _get_sample(self, sample):
        ...