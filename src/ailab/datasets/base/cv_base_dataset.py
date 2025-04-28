from PIL import Image
from torchvision import transforms
from typing import Callable

from ..utils import load_dicts_from_jsonlines

from ._base import AnnotationFileLoadingDataset


std_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda image: image.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.48145466, 0.4578275, 0.40821073],
        std = [0.26862954, 0.26130258, 0.27577711]
    )
])


class ClassificationImageDataset(AnnotationFileLoadingDataset):
    """
    根据 JSONL 格式的标注文件加载图像和标签，每个 sample 是一个字典，由图像路径和图像标签以及其他信息组成。
    Args:
        annotation_file (str): 包含图像路径和标签的 JSONL 文件路径。
        transform (callable): 应用于图像的转换操作。
    """
    def __init__(
        self,
        annotation_file: str,
        x_key: str,
        y_key: str,
        transform: Callable = std_transform
    ):
        super().__init__(annotation_file = annotation_file, load_fun = load_dicts_from_jsonlines)
        self.x_key = x_key
        self.y_key = y_key
        self.transform = transform

    def _get_sample(self, sample: dict) -> tuple:
        """
        根据样本字典加载图像和标签。
        Args:
            sample (dict): 包含 "image_path" 和 "category_id" 的字典。
        Returns:
            tuple: 处理后的图像和标签。
        """
        path = sample[self.x_key]
        label = int(sample[self.y_key])
        image = Image.open(path)
        image = self.transform(image)

        return image, label