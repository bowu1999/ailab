import cv2
import random
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Union, Sequence, Tuple, Dict, List


def draw_bounding_boxes_flexible(
    image: Union[str, Path, np.ndarray, Image.Image],
    bboxes: Sequence[Tuple],
    masks: List[Union[List[List[float]], np.ndarray]] = None,
    keypoints: List[Sequence[Tuple[float, float]]] = None,
    bbox_format: str = 'xyxy',
    output_path: Union[str, Path, None] = None,
    color: Tuple[int,int,int] = (0,255,0),
    thickness: int = 2,
    base_font_scale: float = 0.5,
    base_font_thickness: int = 1,
    label_colors: Dict[str, Tuple[int,int,int]] = None,
    mask_alpha: float = 0.4,
    keypoint_radius: int = 3
) -> np.ndarray:
    """
    在图片上绘制各种格式的边界框，并支持多种输入/输出，以及自动分配颜色：
      - 若提供 label_colors，则针对指定 label 使用对应颜色；
      - 否则相同 label 始终使用相同随机色；
      - 若无 label，则每个框使用不同随机色。
    同时还支持绘制：
      - 边界框 bboxes（各种格式：xyxy/xywh/cxcywh 及相对 rel_…）
      - 分割 masks（COCO 多边形或二值 RLE 数组）
      - 关键点 keypoints（[(x1,y1),(x2,y2),…]）
    
    参数:
      image: 文件路径 | OpenCV 数组 | PIL Image 对象
      bboxes: 每个元素可为:
        - (x_min, y_min, x_max, y_max)
        - (label, conf, x_min, y_min, x_max, y_max)
        - (x_min, y_min, w, h)
        - (label, conf, x_min, y_min, w, h)
        - (x_center, y_center, w, h)  # 相对或绝对
        - (label, conf, x_center, y_center, w, h)
      bbox_format: 支持 'xyxy','xywh','cxcywh', 以及它们的归一化版本 'rel_xyxy','rel_xywh','rel_cxcywh'
      masks: 与 bboxes 一一对应的分割，多边形 [[x0,y0,...],…] 或者二值 mask ndarray
      keypoints: 与 bboxes 一一对应的 keypoints 列表
      output_path: 若提供，则写入磁盘；否则返回数组
      color: 默认颜色（仅在无 label 且随机色未启用时使用）
      thickness: 边框线宽
      base_font_scale: 文字基准大小，按图像高度缩放
      base_font_thickness: 文字基准粗细，按对角线长度缩放
      label_colors: 用户指定的 label→BGR 颜色映射
      mask_alpha: mask 绘制透明度
      keypoint_radius: 关键点半径
    返回:
      带有框的 OpenCV BGR 图像 (ndarray)
    """
    # —————————— 1. 读取/转换为 OpenCV BGR ndarray ——————————
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Cannot read image at {image}")
    elif isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise TypeError("Unsupported image type")
    h, w = img.shape[:2]
    # —————————— 2. 文字参数自适应 ——————————
    font_scale = base_font_scale * (h / 640)
    diag = (w**2 + h**2)**0.5
    font_thickness = max(1, int(base_font_thickness * diag / 1000))
    # —————————— 3. 标签颜色及随机色工具 ——————————
    assigned_colors: Dict[str, Tuple[int,int,int]] = {}
    label_colors = label_colors or {}
    def _rand_color():
        return (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    # —————————— 4. 绘制每个实例 ——————————
    n = len(bboxes)
    for i, bbox in enumerate(bboxes):
        # 4.1 解析 label/conf 与坐标
        if len(bbox) == 4:
            label = None; conf = None; coords = bbox
        elif len(bbox) == 5:
            label, *coords = bbox; conf = None
        elif len(bbox) == 6:
            label, conf, *coords = bbox
        else:
            raise ValueError("每个 bbox 应包含 4、5 或 6 个值")
        # 4.2 决定颜色
        if label is not None:
            s_label = str(label)
            if s_label in label_colors:
                box_color = label_colors[s_label]
            else:
                if s_label not in assigned_colors:
                    assigned_colors[s_label] = _rand_color()
                box_color = assigned_colors[s_label]
        else:
            box_color = _rand_color()
        # 4.3 坐标 → 绝对像素 xyxy
        fmt = bbox_format.lower()
        if fmt in ('xyxy','rel_xyxy'):
            x1,y1,x2,y2 = coords
            if fmt.startswith('rel_'):
                x1,y1,x2,y2 = x1*w, y1*h, x2*w, y2*h
        elif fmt in ('xywh','rel_xywh'):
            x1,y1,bw,bh = coords
            if fmt.startswith('rel_'):
                x1,y1,bw,bh = x1*w, y1*h, bw*w, bh*h
            x2,y2 = x1+bw, y1+bh
        elif fmt in ('cxcywh','rel_cxcywh'):
            cx,cy,bw,bh = coords
            if fmt.startswith('rel_'):
                cx,cy,bw,bh = cx*w, cy*h, bw*w, bh*h
            x1,y1 = cx-bw/2, cy-bh/2
            x2,y2 = x1+bw, y1+bh
        else:
            raise ValueError(f"Unsupported bbox_format: {bbox_format}")
        pt1, pt2 = (int(x1),int(y1)), (int(x2),int(y2))
        # ———— 4.4 绘制 Mask（如果提供） ————
        if masks is not None:
            # masks[i] 是一个 list，每项为一个子 mask 的 [x0,y0,x1,y1,...]
            for poly in masks[i]:
                # 将扁平列表转为 Nx2 点阵
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                # 画到 overlay，然后 blend
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], box_color)
                cv2.addWeighted(overlay, mask_alpha, img, 1-mask_alpha, 0, img)
        # ———— 4.5 绘制边界框 ————
        cv2.rectangle(img, pt1, pt2, box_color, thickness)
        # ———— 4.6 绘制 Label+Conf ————
        if label is not None:
            text = f"{label}: {conf:.2f}" if conf is not None else str(label)
            # 测量文字尺寸
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            # 原先背景框（放在框外）
            bg_tl = (pt1[0], pt1[1] - th - 4)
            bg_br = (pt1[0] + tw + 4, pt1[1])
            # 如果背景框上边在图外（y < 0），则背景框改为框内顶部
            if bg_tl[1] < 0:
                # 放在边界框内部：左上角点下移 th + 4
                bg_tl = (pt1[0], pt1[1] + 4)
                bg_br = (pt1[0] + tw + 4, pt1[1] + th + 4)
                text_org = (pt1[0] + 2, pt1[1] + th + 2)
            else:
                text_org = (pt1[0] + 2, pt1[1] - 2)
            # 绘制背景
            cv2.rectangle(img, bg_tl, bg_br, box_color, -1)
            # 绘制文字
            cv2.putText(img, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness)
        # ———— 4.7 绘制 Keypoints（如果提供） ————
        if keypoints is not None:
            for (kx, ky) in keypoints[i]:
                if 0 <= kx <= 1 and 0 <= ky <= 1:
                    kx, ky = kx * w, ky * h
                cv2.circle(img, (int(kx),int(ky)), keypoint_radius, box_color, -1)

    # —————————— 5. 保存或返回 ——————————
    if output_path:
        cv2.imwrite(str(output_path), img)
    return img


def display_image_from_path(image_path):
    """
    从给定的路径读取图像并使用Matplotlib显示。
    
    参数:
    image_path (str): 图像文件的路径。
    """
    # 使用OpenCV读取图像
    image_bgr = cv2.imread(image_path)

    # 检查图片是否加载成功
    if image_bgr is None:
        print("Error: Could not load image.")
        return
    
    # 将BGR格式转换为RGB格式
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 使用Matplotlib显示图像
    plt.figure(figsize=(10, 6))  # 可选：设置图像大小
    plt.imshow(image_rgb)
    plt.title('Image loaded with OpenCV and displayed using Matplotlib')
    plt.axis('off')  # 不显示坐标轴
    plt.show()


def display_image_from_cv2(image):
    """
    从给定的路径读取图像并使用Matplotlib显示。
    
    参数:
    image_path (str): 图像文件的路径。
    """
    # 将BGR格式转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用Matplotlib显示图像
    plt.figure(figsize=(10, 6))  # 可选：设置图像大小
    plt.imshow(image_rgb)
    plt.title('Image loaded with OpenCV and displayed using Matplotlib')
    plt.axis('off')  # 不显示坐标轴
    plt.show()