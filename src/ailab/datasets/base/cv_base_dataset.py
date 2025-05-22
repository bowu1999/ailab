import cv2
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Callable, Any, List, Union
import os

# 禁用 Albumentations 的在线版本检查
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'



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


# 定义数据增强流程
common_aug = A.Compose(
    [
        # 将图像的最长边缩放到不超过640，保持宽高比
        A.LongestMaxSize(max_size=640),
        # 如果图像尺寸小于640x640，则进行填充，使用常数值填充（黑色）
        A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        # 以50%的概率随机裁剪512x512的区域
        A.RandomCrop(height=512, width=512, p=0.5),
        # 以50%的概率进行水平翻转
        A.HorizontalFlip(p=0.5),
        # 以20%的概率进行垂直翻转
        A.VerticalFlip(p=0.2),
        # 以30%的概率进行仿射变换，包括平移、缩放和旋转
        A.Affine(
            translate_percent=0.05,  # 平移范围为±5%
            scale=(0.9, 1.1),        # 缩放范围为90%到110%
            rotate=(-5, 5),          # 旋转范围为±5度
            p=0.3
        ),
        # 以80%的概率随机调整亮度和对比度
        # brightness_limit和contrast_limit控制调整的范围
        # 增加这些值可以增强模型对光照变化的鲁棒性
        A.RandomBrightnessContrast(
            brightness_limit=0.3,  # 亮度调整范围为±30%
            contrast_limit=0.3,    # 对比度调整范围为±30%
            p=0.8
        ),
        # 以80%的概率随机调整色调、饱和度和明度
        # hue_shift_limit控制色调的变化范围
        # sat_shift_limit控制饱和度的变化范围
        # val_shift_limit控制明度的变化范围
        # 增加这些值可以增强模型对颜色变化的鲁棒性
        A.HueSaturationValue(
            hue_shift_limit=15,    # 色调调整范围为±15
            sat_shift_limit=25,    # 饱和度调整范围为±25
            val_shift_limit=25,    # 明度调整范围为±25
            p=0.8
        ),
        # 以50%的概率应用高斯模糊
        # blur_limit控制模糊核的大小范围，必须为奇数
        # 增加blur_limit的上限可以增强模型对模糊图像的鲁棒性
        A.GaussianBlur(
            blur_limit=(3, 9),     # 模糊核大小范围为3到9
            p=0.5
        ),
        # 标准化图像，使其均值为0，标准差为1
        # 使用ImageNet的均值和标准差
        A.Normalize(
            mean=(0.485, 0.456, 0.406),  # RGB通道的均值
            std=(0.229, 0.224, 0.225)    # RGB通道的标准差
        ),
        # 将图像重新调整为640x640
        A.Resize(height=640, width=640, interpolation=cv2.INTER_LINEAR, p=1.0),
        # 将图像转换为PyTorch张量
        ToTensorV2()
    ],
    # 设置边界框的参数
    bbox_params=A.BboxParams(
        format='yolo',             # 边界框格式为YOLO格式
        label_fields=['category_ids'],  # 标签字段
        min_visibility=0.5         # 最小可见度阈值，过滤掉可见度低的边界框
    ),
    # 设置关键点的参数
    keypoint_params=A.KeypointParams(
        format='xy',               # 关键点格式为(x, y)
        remove_invisible=False     # 不移除不可见的关键点
    )
)

def letterbox(img, new_size=640, color=(114, 114, 114)):
    """等比缩放并填充到 (new_size, new_size)，返回 img, (scale, pad_w, pad_h)"""
    h0, w0 = img.shape[:2]
    r = min(new_size / h0, new_size / w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]
    dw /= 2
    dh /= 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, (r, left, top)

class COCODataset:
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
        annotation_file: str,
        images_root: str,
        img_size: int = 640,
        tasks: list = ['bbox'],
        box_format: str = 'yolo',
        min_area: float = 10.0,
        ignore_crowd: bool = True,
        transforms_img=common_aug,
        transforms_target=None,
        multiscale: bool = False,
        mixup_mosaic: bool = False,
        cache_images: bool = False,
        return_orig: bool = False
    ):
        self.images_root = Path(images_root)
        self.img_size = img_size
        self.tasks = set(tasks)
        self.box_format = box_format
        self.min_area = min_area
        self.ignore_crowd = ignore_crowd
        self.transforms_img = transforms_img
        self.transforms_target = transforms_target
        self.multiscale = multiscale
        self.mixup_mosaic = mixup_mosaic
        self.cache_images = cache_images
        self.return_orig = return_orig

        with open(annotation_file) as f:
            data = json.load(f)
        self.cat_map = {c['id']: i for i, c in enumerate(data['categories'])}
        ann_index = {}
        for img in data['images']:
            ann_index[img['id']] = []
        for ann in data['annotations']:
            if self.ignore_crowd and ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if w * h < self.min_area:
                continue
            ann_index[ann['image_id']].append(ann)

        self.samples = []
        for img in data['images']:
            anns = ann_index[img['id']]
            if not anns:
                continue
            self.samples.append({
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height'],
                'anns': anns
            })

        if cache_images:
            self._cache = [self._load_image(s) for s in self.samples]
        else:
            self._cache = [None] * len(self.samples)

    def __len__(self):
        return len(self.samples)

    def _load_image(self, sample):
        path = self.images_root / sample['file_name']
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot load {path}")
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]

        size = (np.random.choice([self.img_size, self.img_size // 2, self.img_size * 2])
                if self.multiscale else self.img_size)

        img0 = self._cache[idx] if self.cache_images else self._load_image(sample)
        h0, w0 = img0.shape[:2]

        # 获取原始边界框
        bboxes, class_ids, masks, keypoints = [], [], [], []
        for ann in sample['anns']:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, w, h])
            class_ids.append(self.cat_map[ann['category_id']])
            if 'segmentation' in ann and 'mask' in self.tasks:
                mask = np.zeros((h0, w0), dtype=np.uint8)
                for poly in ann['segmentation']:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                masks.append(mask)
            if 'keypoints' in ann and 'keypoints' in self.tasks:
                pts = np.array(ann['keypoints']).reshape(-1, 3)[:, :2]
                keypoints.append(pts.tolist())

        # 应用 letterbox
        img, (r, pad_w, pad_h) = letterbox(img0, size, color=(114, 114, 114))

        # 调整边界框坐标
        bboxes = np.array(bboxes)
        bboxes[:, 0] = bboxes[:, 0] * r + pad_w  # x
        bboxes[:, 1] = bboxes[:, 1] * r + pad_h  # y
        bboxes[:, 2] = bboxes[:, 2] * r          # w
        bboxes[:, 3] = bboxes[:, 3] * r          # h

        # 转换为 YOLO 格式（cx, cy, w, h）并归一化
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2  # cx
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2  # cy
        bboxes[:, 0] /= size
        bboxes[:, 1] /= size
        bboxes[:, 2] /= size
        bboxes[:, 3] /= size

        aug_kwargs = dict(image=img, bboxes=bboxes.tolist(), category_ids=class_ids)
        if masks:
            aug_kwargs['masks'] = masks
        if keypoints:
            aug_kwargs['keypoints'] = keypoints
        aug = self.transforms_img(**aug_kwargs)

        img_t = aug['image']
        target = {}
        if 'bbox' in self.tasks:
            boxes_out = np.array(aug['bboxes'], dtype=np.float32)
            if self.box_format == 'pascal':
                xy = boxes_out[:, :2] - boxes_out[:, 2:] / 2
                wh = boxes_out[:, 2:]
                boxes_out = np.concatenate([xy, xy + wh], axis=1)
            elif self.box_format == 'coco':
                boxes_out = np.concatenate([
                    (boxes_out[:, :2] - boxes_out[:, 2:] / 2) * [size, size],
                    boxes_out[:, 2:] * [size, size]
                ], axis=1)
            target['boxes'] = torch.tensor(boxes_out)
            target['labels'] = torch.tensor(aug['category_ids'], dtype=torch.long)

        if 'mask' in self.tasks and masks:
            target['masks'] = torch.stack([torch.tensor(m, dtype=torch.uint8) for m in aug['masks']])

        if 'keypoints' in self.tasks and keypoints:
            target['keypoints'] = torch.tensor(aug['keypoints'], dtype=torch.float32)

        if self.transforms_target:
            pil = Image.fromarray(cv2.cvtColor(img_t.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB))
            pil, target = self.transforms_target(pil, target)
            img_t = torch.from_numpy(np.array(pil)[:, :, ::-1]).permute(2, 0, 1)
        
        if self.return_orig and 'bbox' in self.tasks:
            orig_boxes = []
            for ann in sample['anns']:
                # COCO 原始格式 [xmin, ymin, w, h]
                x, y, w, h = ann['bbox']
                # 转成 [x1, y1, x2, y2]
                orig_boxes.append([x, y, x + w, y + h])
            target['orig_boxes'] = torch.tensor(orig_boxes, dtype=torch.float32)

        return {'image': img_t, 'target': target}
    
    def get_vis_sample(self, idx, apply_aug=True, show_boxes=True):
        """
        获取用于可视化的样本图像及其标注信息。
        
        参数：
            idx (int): 样本索引。
            apply_aug (bool): 是否应用数据增强。
            show_boxes (bool): 是否在图像上绘制边界框。
        
        返回：
            vis_np (np.ndarray): 可视化图像（RGB格式）。
            vis_info (dict): 包含边界框和标签的字典。
        """
        def denormalize(image, mean, std):
            mean = np.array(mean).reshape(3, 1, 1)
            std = np.array(std).reshape(3, 1, 1)
            return (image * std + mean).clip(0, 1)

        # 获取归一化后的图像和目标信息
        sample = self.__getitem__(idx)
        img_t = sample['image']  # Tensor, shape: (3, H, W)
        target = sample['target']
        boxes = target['boxes'].cpu().numpy()  # shape: (N, 4)
        labels = target['labels'].cpu().numpy()

        # 获取原始图像尺寸
        sample_meta = self.samples[idx]
        h0, w0 = sample_meta['height'], sample_meta['width']

        # 反归一化图像
        vis_np = img_t.cpu().numpy()
        vis_np = denormalize(vis_np, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        vis_np = np.transpose(vis_np, (1, 2, 0))  # CHW -> HWC
        vis_np = (vis_np * 255).astype(np.uint8)

        # 将图像从BGR转换为RGB（如果需要）
        vis_np = cv2.cvtColor(vis_np, cv2.COLOR_BGR2RGB)

        # 绘制边界框
        if show_boxes and boxes.size > 0:
            for box in boxes:
                if self.box_format == 'yolo':
                    # YOLO格式: [cx, cy, w, h]，归一化
                    cx, cy, w, h = box
                    x1 = int((cx - w / 2) * vis_np.shape[1])
                    y1 = int((cy - h / 2) * vis_np.shape[0])
                    x2 = int((cx + w / 2) * vis_np.shape[1])
                    y2 = int((cy + h / 2) * vis_np.shape[0])
                elif self.box_format == 'pascal':
                    # Pascal VOC格式: [xmin, ymin, xmax, ymax]
                    x1, y1, x2, y2 = box.astype(int)
                elif self.box_format == 'coco':
                    # COCO格式: [xmin, ymin, w, h]
                    x1, y1, w_box, h_box = box
                    x2 = int(x1 + w_box)
                    y2 = int(y1 + h_box)
                    x1, y1 = int(x1), int(y1)
                else:
                    raise ValueError(f"Unsupported box format: {self.box_format}")
                cv2.rectangle(vis_np, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        return vis_np, {'boxes': boxes, 'labels': labels}
