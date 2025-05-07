import warnings
import pandas as pd
from pathlib import Path
from PIL import ImageFile
# 允许 Pillow 在遇到截断图像时尽量加载
ImageFile.LOAD_TRUNCATED_IMAGES = True
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import List, Tuple, Union


class LabBaseDataset(Dataset, ABC):
    """
    核心抽象基类：自动容错，跳过加载或转换失败的样本。
    子类只需实现 _get_raw 和 _get_sample。
    """
    def __len__(self):
        return self._len()

    def __getitem__(self, idx):
        if self._len() == 0:
            raise RuntimeError("Dataset is empty!")
        # 尝试最多取 len 次，跳过坏样本
        for _ in range(self._len()):
            raw = self._get_raw(idx)
            try:
                sample = self._get_sample(raw)
                if not isinstance(sample, dict):
                    raise TypeError(f"_get_sample must return dict, got {type(sample)}")
                return sample
            except Exception as e:
                warnings.warn(f"[Dataset] idx={idx} raw={raw} load failed: {e}. Skipped.")
                idx = (idx + 1) % self._len()
        # 所有样本都失败
        raise RuntimeError("All samples failed to load or transform.")

    @abstractmethod
    def _get_raw(self, idx):
        ...

    @abstractmethod
    def _len(self):
        ...

    @abstractmethod
    def _get_sample(self, raw):
        ...


# —— 标注存储方式抽象类 ——
class FileAnnotationDataset(LabBaseDataset):
    """
    单文件存储标注，如 CSV、TXT、JSON 列表等。
    load_fun 返回行列表或 DataFrame。
    """
    def __init__(self, annotation_file: Union[str, Path, List[str]], load_fun):
        if isinstance(annotation_file, (str, Path)):
            annotation_paths = [annotation_file]
        else:
            annotation_paths = annotation_file
        self.samples = []
        for annotation_file in annotation_paths:
            self.samples += load_fun(annotation_file)

    def _len(self):
        return len(self.samples)

    def _get_raw(self, idx):
        return self.samples[idx]

    @abstractmethod
    def _get_sample(self, raw):
        """子类实现：raw 可能是 str 行、dict、Series 等"""
        ...


class FolderPerClassDataset(LabBaseDataset):
    """
    每个子文件夹表示一个类别，常用于图像分类。
    """
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.samples = []  # list of (path, class_idx)
        for cls in self.classes:
            for img in (self.root/cls).iterdir():
                self.samples.append((img, self.classes.index(cls)))

    def _len(self):
        return len(self.samples)

    def _get_raw(self, idx):
        return self.samples[idx]  # (path, class_idx)

    @abstractmethod
    def _get_sample(self, raw):
        """raw = (path, class_idx)"""
        ...


class TableAnnotationDataset(LabBaseDataset):
    """
    标注存储于表格，如 CSV/Excel。
    """
    def __init__(self, table_path: str, loader_fun=pd.read_csv):
        self.table = loader_fun(table_path)

    def _len(self):
        return len(self.table)

    def _get_raw(self, idx):
        return self.table.iloc[idx]

    @abstractmethod
    def _get_sample(self, raw):
        """raw 是 pandas Series"""
        ...


class CocoAnnotationDataset(LabBaseDataset):
    """
    COCO 格式 JSON 标注，使用 pycocotools 加载。
    raw -> (image_id, image_info, annotations)
    """
    def __init__(self, coco_json: str, images_dir: str):
        from pycocotools.coco import COCO
        self.coco = COCO(coco_json)
        self.img_dir = Path(images_dir)
        self.ids = list(self.coco.imgs.keys())

    def _len(self):
        return len(self.ids)

    def _get_raw(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        ann = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        return img_id, info, ann

    @abstractmethod
    def _get_sample(self, raw):
        """子类实现：raw = (id, img_info, ann_list)"""
        ...


class VocAnnotationDataset(LabBaseDataset):
    """
    Pascal VOC XML 格式标注。
    raw -> (image_path, xml_dict)
    """
    def __init__(self, images_dir: str, annotations_dir: str, parser_fun):
        self.img_dir = Path(images_dir)
        self.ann_dir = Path(annotations_dir)
        self.parser = parser_fun  # 解析 XML -> dict
        self.names = [p.stem for p in self.ann_dir.glob('*.xml')]

    def _len(self):
        return len(self.names)

    def _get_raw(self, idx):
        name = self.names[idx]
        img_p = self.img_dir/f"{name}.jpg"
        xml_p = self.ann_dir/f"{name}.xml"
        xml_dict = self.parser(str(xml_p))
        return img_p, xml_dict

    @abstractmethod
    def _get_sample(self, raw):
        """raw = (image_path, parsed_xml)"""
        ...


# —— 任务类型抽象类 ——
class ClassificationTask(ABC):
    """
    图像/文本/表格 分类子任务。
    子类需实现 load_input, load_label。
    """
    @abstractmethod
    def load_input(self, raw):
        ...

    @abstractmethod
    def load_label(self, raw):
        ...

    def _get_sample(self, raw):
        x = self.load_input(raw)
        y = self.load_label(raw)
        return {'input': x, 'label': y}


class RegressionTask(ABC):
    """回归子任务"""
    @abstractmethod
    def load_input(self, raw):
        ...

    @abstractmethod
    def load_target(self, raw):
        ...

    def _get_sample(self, raw):
        return {'input': self.load_input(raw),
                'target': self.load_target(raw)}


class DetectionTask(ABC):
    """目标检测子任务"""
    @abstractmethod
    def load_image(self, raw):
        ...

    @abstractmethod
    def load_boxes(self, raw):
        ...

    @abstractmethod
    def load_labels(self, raw):
        ...

    def _get_sample(self, raw):
        return {'image': self.load_image(raw),
                'boxes': self.load_boxes(raw),
                'labels': self.load_labels(raw)}


class SegmentationTask(ABC):
    """语义分割子任务"""
    @abstractmethod
    def load_image(self, raw):
        ...

    @abstractmethod
    def load_mask(self, raw):
        ...

    def _get_sample(self, raw):
        return {'image': self.load_image(raw),
                'mask': self.load_mask(raw)}


class KeypointTask(ABC):
    """关键点检测子任务"""
    @abstractmethod
    def load_image(self, raw):
        ...

    @abstractmethod
    def load_keypoints(self, raw):
        ...

    def _get_sample(self, raw):
        return {'image': self.load_image(raw),
                'keypoints': self.load_keypoints(raw)}


class PanopticTask(ABC):
    """全景分割子任务"""
    @abstractmethod
    def load_image(self, raw):
        ...

    @abstractmethod
    def load_semantic(self, raw):
        ...

    @abstractmethod
    def load_instance(self, raw):
        ...

    def _get_sample(self, raw):
        return {'image': self.load_image(raw),
                'semantic': self.load_semantic(raw),
                'instance': self.load_instance(raw)}


# 更多任务抽象：
class VideoClassificationTask(ABC):
    @abstractmethod
    def load_frames(self, raw): ...
    @abstractmethod
    def load_label(self, raw): ...
    def _get_sample(self, raw): return {'frames': self.load_frames(raw), 'label': self.load_label(raw)}

class TimeSeriesTask(ABC):
    @abstractmethod
    def load_series(self, raw): ...
    @abstractmethod
    def load_target(self, raw): ...
    def _get_sample(self, raw): return {'series': self.load_series(raw), 'target': self.load_target(raw)}

class AudioClassificationTask(ABC):
    @abstractmethod
    def load_audio(self, raw): ...
    @abstractmethod
    def load_label(self, raw): ...
    def _get_sample(self, raw): return {'audio': self.load_audio(raw), 'label': self.load_label(raw)}

class SpeechRecognitionTask(ABC):
    @abstractmethod
    def load_audio(self, raw): ...
    @abstractmethod
    def load_transcript(self, raw): ...
    def _get_sample(self, raw): return {'audio': self.load_audio(raw), 'text': self.load_transcript(raw)}

class TextClassificationTask(ABC):
    @abstractmethod
    def load_text(self, raw): ...
    @abstractmethod
    def load_label(self, raw): ...
    def _get_sample(self, raw): return {'text': self.load_text(raw), 'label': self.load_label(raw)}

class Seq2SeqTask(ABC):
    @abstractmethod
    def load_source(self, raw): ...
    @abstractmethod
    def load_target(self, raw): ...
    def _get_sample(self, raw): return {'source': self.load_source(raw), 'target': self.load_target(raw)}

class GraphTask(ABC):
    @abstractmethod
    def load_graph(self, raw): ...
    @abstractmethod
    def load_label(self, raw): ...
    def _get_sample(self, raw): return {'graph': self.load_graph(raw), 'label': self.load_label(raw)}


# —— 用户示例 ——
# 自定义数据集示例：
# class MyImageClasData(FolderPerClassDataset, ClassificationTask):
#     def __init__(self, root):
#         FolderPerClassDataset.__init__(self, root)
#     def load_input(self, raw):
#         path, _ = raw; return read_image(path)
#     def load_label(self, raw):
#         _, cls = raw; return cls
#
# class MyCocoDetData(CocoAnnotationDataset, DetectionTask):
#     def __init__(self, json, img_dir):
#         CocoAnnotationDataset.__init__(self, json, img_dir)
#     def load_image(self, raw):
#         img_id, info, ann = raw; return load_img(self.img_dir/info['file_name'])
#     def load_boxes(self, raw):
#         _, _, ann = raw; return [x['bbox'] for x in ann]
#     def load_labels(self, raw):
#         _, _, ann = raw; return [x['category_id'] for x in ann]

# 用户可自由继承 LabBaseDataset 或任意组合上述抽象类，自行扩展特殊场景。