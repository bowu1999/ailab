import json
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
from typing import Callable, Any, List, Union

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


class COCODataset(FileAnnotationDataset):
    """
    annotation_file: COCO JSON 文件路径（或表）
    images_root: 图像根目录
    tasks: 任务列表，支持：'bbox'、'mask'、'segmentation'、 'keypoints'
    box_format: 边界框格式，支持：
        'yolo'（归一化中心，长宽形式）、'pascal'（[xmin,ymin,xmax,ymax]）、'coco'（原始 [xmin,ymin,w,h]）
    min_area: 过滤小于阈值的边界框（像素面积）
    ignore_crowd: 是否跳过 iscrowd=1 的注释
    transforms_img: 图像级增强（如色彩抖动、随机裁剪）
    transforms_target: 目标（boxes/masks/keypoints）级增强钩子
    multiscale: 是否启用多尺度训练：在取样时随机调整 img_size
    mixup_mosaic: 占位参数，后续可接入 MixUp/MosaicAugmenter
    cache_images: 是否缓存所有图像到内存（适用于小数据集）
    """
    def __init__(
        self,
        annotation_file,
        images_root: str,
        tasks: list = ['bbox'], # 支持 'bbox','mask','segmentation','keypoints'
        box_format: str = 'yolo', # 'yolo' | 'pascal' | 'coco'
        min_area: float = 10.0,
        ignore_crowd: bool = True,
        transforms_img = None, # 图像级变换
        transforms_target = None, # 目标级变换回调
        multiscale: bool = False, # 多尺度训练
        mixup_mosaic: bool = False, # MixUp/Mosaic 占位
        cache_images: bool = False
    ):
        """
        annotation_file: COCO JSON 文件或文件列表
        images_root: 图像存放根目录
        """
        self.images_root = Path(images_root)
        self.tasks = tasks
        self.box_format = box_format
        self.min_area = min_area
        self.ignore_crowd = ignore_crowd
        self.transforms_img = transforms_img
        self.transforms_target = transforms_target
        self.multiscale = multiscale
        self.mixup_mosaic = mixup_mosaic
        self.cache_images = cache_images
        # 解析 COCO JSON，得到 raw 样本列表
        raws = self._parse_coco(annotation_file)
        super().__init__(raws, load_fun=lambda x: raws)  # 利用父类加载 samples
        # 可选：缓存图像到内存
        if cache_images:
            self._cache = [self._load_image(r) for r in raws]
        else:
            self._cache = [None] * len(self.samples)

    def _parse_coco(self, json_path):
        data = json.load(open(json_path))
        # 连续类别映射
        cat_map = {c['id']: i for i, c in enumerate(data['categories'])}
        # 注释按 image_id 分组
        ann_index = defaultdict(list)
        for ann in data['annotations']:
            if self.ignore_crowd and ann.get('iscrowd', 0):
                continue
            if ann['bbox'][2] * ann['bbox'][3] < self.min_area:
                continue  # 小面积过滤
            ann_index[ann['image_id']].append(ann)
        # 生成 raw 列表
        raws = []
        for img in data['images']:
            imgs_anns = ann_index.get(img['id'], [])
            if not imgs_anns:
                continue
            raws.append({
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height'],
                'anns': imgs_anns,
                'cat_map': cat_map
            })
        return raws

    def _load_image(self, raw):
        path = self.images_root / raw['file_name']
        return Image.open(path).convert('RGB')

    def _get_raw(self, idx):
        return self.samples[idx]

    def _len(self):
        return len(self.samples)

    def _get_sample(self, raw):
        # 加载或取缓存图像
        idx = self.samples.index(raw)
        img = self._cache[idx] if self.cache_images else self._load_image(raw)
        w, h = raw['width'], raw['height']
        # 可选多尺度
        if self.multiscale:
            new_size = np.random.choice([320, 416, 512, 608])
            img = img.resize((new_size, new_size), Image.BILINEAR)
            scale_x, scale_y = new_size / w, new_size / h
        else:
            scale_x = scale_y = 1.0
        # 标签
        target = {}
        # 1. 边界框
        if 'bbox' in self.tasks:
            boxes = []
            for ann in raw['anns']:
                x, y, bw, bh = ann['bbox']
                # 转为所需格式
                # 采用归一化的中心坐标和宽高，值域在 [0,1] 之间
                if self.box_format == 'yolo':
                    cx = (x + bw/2) / w
                    cy = (y + bh/2) / h
                    boxes.append([raw['cat_map'][ann['category_id']], cx, cy, bw/w, bh/h])
                # 使用绝对像素的左上角和右下角坐标 [x_min, y_min, x_max, y_max]
                elif self.box_format == 'pascal':
                    boxes.append([x, y, x+bw, y+bh, raw['cat_map'][ann['category_id']]])
                # 使用绝对像素的左上角坐标加上宽高 [x_min, y_min, w, h]
                else:  # coco
                    boxes.append([x, y, bw, bh, raw['cat_map'][ann['category_id']]])
            target['boxes'] = np.array(boxes, dtype=np.float32)
        # 2. Mask 多边形或 RLE
        if 'mask' in self.tasks and 'segmentation' in raw['anns'][0]:
            masks = [ann['segmentation'] for ann in raw['anns']]
            target['masks'] = masks  # 用户可在 transforms_target 中进一步处理
        # 3. 关键点
        if 'keypoints' in self.tasks and 'keypoints' in raw['anns'][0]:
            kps = [ann['keypoints'] for ann in raw['anns']]
            target['keypoints'] = kps
        # 应用用户定义的目标变换
        if self.transforms_target:
            img, target = self.transforms_target(img, target)
        # 应用图像级变换
        if self.transforms_img:
            img = self.transforms_img(img)

        return {'image': img, 'target': target}