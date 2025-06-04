import cv2
import torch
import random
import colorsys
import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib import patches
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


def visualize_masks(dataset, idx, alpha=0.5, show_boxes=False, figsize=(12, 8)):
    """
    可视化分割数据集中的图像和掩码
    
    Args:
        dataset: COCOSegmentationDataset实例
        idx: 样本索引
        alpha: mask透明度 (0-1)
        show_boxes: 是否显示边界框（分割数据集通常不需要）
        figsize: 图像大小
    """
    # 获取数据
    sample = dataset[idx]
    img_tensor = sample['image']
    # 检查是语义分割还是实例分割
    if dataset.mode == 'semantic':
        label = sample['label']
        target = None
    else:
        target = sample['target']
        label = None
    # 处理图像（不需要反归一化，因为数据集已经返回0-1范围）
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img)
    if dataset.mode == 'semantic':
        # 语义分割模式
        label_np = label.cpu().numpy()
        # 创建彩色overlay
        h, w = label_np.shape
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        # 为每个类别分配颜色
        for class_id in range(dataset.num_classes):
            mask = label_np == class_id
            if mask.any():
                color = plt.cm.tab20(class_id / dataset.num_classes)[:3]
                overlay[mask] = color
        # 叠加显示
        ax.imshow(overlay, alpha=alpha)
    else:
        # 实例分割模式
        if 'masks' in target and len(target['masks']) > 0:
            masks = target['masks'].cpu().numpy()
            labels = target['labels'].cpu().numpy()
            # 创建彩色overlay
            h, w = img.shape[:2]
            overlay = np.zeros((h, w, 3), dtype=np.float32)
            # 为每个实例生成颜色
            num_instances = len(masks)
            for i, (mask, label) in enumerate(zip(masks, labels)):
                hue = i / max(num_instances, 1)
                color = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
                # 应用mask
                for c in range(3):
                    overlay[:, :, c] = np.maximum(overlay[:, :, c], mask * color[c])
            # 叠加显示
            ax.imshow(overlay, alpha=alpha)
            # 添加标签
            for i, (mask, label) in enumerate(zip(masks, labels)):
                if mask.sum() > 0:
                    # 找到mask的中心
                    y_indices, x_indices = np.where(mask > 0)
                    cy = int(np.mean(y_indices))
                    cx = int(np.mean(x_indices))
                    # 获取类别名称
                    if hasattr(dataset, 'categories'):
                        cat_name = next(
                            (c['name'] for c in dataset.categories if dataset.cat_map[c['id']] == label),
                            f"Class {label}"
                        )
                    else:
                        cat_name = f"Class {label}"
                    ax.text(cx, cy, cat_name, color='white',
                           bbox=dict(facecolor='black', alpha=0.5),
                           fontsize=8, ha='center', va='center')
    ax.axis('off')
    plt.title(f'Sample {idx} - {dataset.mode.capitalize()} Mode')
    plt.tight_layout()
    plt.show()
    
    # 返回统计信息
    info = {
        'image_shape': img.shape,
        'mode': dataset.mode
    }
    
    if dataset.mode == 'semantic':
        unique_labels = torch.unique(label)
        info['num_classes_present'] = len(unique_labels[unique_labels != dataset.ignore_label])
    else:
        info['num_instances'] = len(target['masks']) if target else 0
    
    return info


def tensor_to_cv2_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        将一个标准化后的PyTorch张量图像还原为OpenCV格式。
        
        参数:
        tensor : torch.Tensor
            输入的标准化后的图像张量，形状应为[C, H, W]。
        mean : list of float, optional
            标准化时使用的平均值，默认为ImageNet的平均值。
        std : list of float, optional
            标准化时使用标准差，默认为ImageNet的标准差。
            
        返回:
        img : numpy.ndarray
            还原后的OpenCV格式图像，形状为[H, W, C]，类型为uint8。
        """
        # 将tensor移到cpu并转换为numpy数组
        tensor = tensor.clone().detach().cpu()
        
        # 去掉批处理维度，如果存在的话
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # 反归一化：乘以标准差并加上平均值
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        
        # 如果最大值小于等于1，则说明还在[0, 1]范围内，需要缩放到[0, 255]
        if tensor.max() <= 1:
            tensor = tensor * 255
        
        # 将tensor转换为numpy数组，并调整通道顺序（C, H, W -> H, W, C）
        img = tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        
        # 如果图像是灰度图（单通道），则不需要转换颜色空间
        if img.shape[2] == 3:
            # OpenCV使用BGR格式，所以需要转换RGB到BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img


def process_segmentation_output(output, original_image, alpha=0.5, colormap='tab20', return_colored_mask=False):
    """
    将语义分割模型的输出处理成mask并覆盖到原图
    
    Args:
        output: 模型输出，形状为 (B, num_classes, H, W)
        original_image: 原始图像，可以是:
            - numpy array (H, W, 3) 
            - torch tensor (3, H, W) 或 (H, W, 3)
        alpha: mask的透明度
        colormap: 颜色映射方案
        return_colored_mask: 是否返回彩色mask
    
    Returns:
        overlayed_image: 叠加了mask的图像
        (可选) colored_mask: 彩色mask
    """
    # 处理输出
    if output.dim() == 4:
        output = output[0]  # 移除batch维度，变成 (80, 512, 1024)
    
    # 获取每个像素的类别（argmax）
    pred_mask = output.argmax(dim=0)  # (512, 1024)
    pred_mask = pred_mask.cpu().numpy()
    
    # 处理原始图像
    if isinstance(original_image, torch.Tensor):
        if original_image.shape[0] == 3:  # (3, H, W)
            original_image = original_image.permute(1, 2, 0)
        original_image = original_image.cpu().numpy()
    
    # 确保图像是0-1范围或0-255范围
    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    else:
        original_image = original_image.astype(np.uint8)
    
    # 调整原图尺寸以匹配mask
    h_mask, w_mask = pred_mask.shape
    if original_image.shape[:2] != (h_mask, w_mask):
        original_image = cv2.resize(original_image, (w_mask, h_mask), interpolation=cv2.INTER_LINEAR)
    
    # 创建彩色mask
    colored_mask = create_colored_mask(pred_mask, num_classes=output.shape[0], colormap=colormap)
    
    # 叠加mask到原图
    overlayed = overlay_mask_on_image(original_image, colored_mask, alpha=alpha)
    
    if return_colored_mask:
        return overlayed, colored_mask
    return overlayed


def create_colored_mask(mask, num_classes=80, colormap='tab20'):
    """
    为每个类别创建不同颜色的mask
    
    Args:
        mask: 类别mask，形状为 (H, W)
        num_classes: 总类别数
        colormap: matplotlib的colormap名称
    
    Returns:
        colored_mask: RGB彩色mask，形状为 (H, W, 3)
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 获取colormap
    if colormap == 'custom':
        # 自定义颜色列表（更鲜艳的颜色）
        colors = generate_distinct_colors(num_classes)
    else:
        cmap = plt.cm.get_cmap(colormap)
        colors = [cmap(i / max(num_classes-1, 1))[:3] for i in range(num_classes)]
    
    # 为每个类别分配颜色
    unique_labels = np.unique(mask)
    for label in unique_labels:
        if label < num_classes:  # 确保标签在范围内
            color = (np.array(colors[label]) * 255).astype(np.uint8)
            colored_mask[mask == label] = color
    
    return colored_mask


def generate_distinct_colors(n):
    """生成n个视觉上易区分的颜色"""
    import colorsys
    colors = []
    for i in range(n):
        hue = i / n
        # 跳过接近黄色的颜色（因为在白色背景上不明显）
        if 0.15 < hue < 0.25:
            hue += 0.1
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors


def overlay_mask_on_image(image, colored_mask, alpha=0.5):
    """
    将彩色mask叠加到图像上
    
    Args:
        image: 原始图像 (H, W, 3)
        colored_mask: 彩色mask (H, W, 3)
        alpha: 透明度
    
    Returns:
        overlayed: 叠加后的图像
    """
    # 确保数据类型一致
    if image.dtype != colored_mask.dtype:
        colored_mask = colored_mask.astype(image.dtype)
    
    # 叠加
    overlayed = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    
    return overlayed


def visualize_segmentation_result(output, original_image, class_names=None, figsize=(15, 5)):
    """
    可视化语义分割结果
    
    Args:
        output: 模型输出 (1, num_classes, H, W)
        original_image: 原始图像
        class_names: 类别名称列表（可选）
        figsize: 图像大小
    """
    # 处理输出
    overlayed, colored_mask = process_segmentation_output(
        output, original_image, alpha=0.5, return_colored_mask=True
    )
    
    # 获取预测的类别
    if output.dim() == 4:
        output = output[0]
    pred_mask = output.argmax(dim=0).cpu().numpy()
    unique_classes = np.unique(pred_mask)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 显示原图
    if isinstance(original_image, torch.Tensor):
        if original_image.shape[0] == 3:
            original_image = original_image.permute(1, 2, 0)
        original_image = original_image.cpu().numpy()
    if original_image.max() <= 1:
        original_image = (original_image * 255).astype(np.uint8)
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 显示彩色mask
    axes[1].imshow(colored_mask)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # 显示叠加结果
    axes[2].imshow(overlayed)
    axes[2].set_title('Overlayed Result')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 打印类别信息
    print(f"Detected {len(unique_classes)} classes: {unique_classes.tolist()}")
    if class_names:
        print("Classes present:")
        for cls in unique_classes:
            if cls < len(class_names):
                print(f"  - Class {cls}: {class_names[cls]}")
    
    plt.show()
    
    return overlayed


# 批量处理函数
def process_batch_segmentation(outputs, images, alpha=0.5):
    """
    批量处理语义分割输出
    
    Args:
        outputs: 模型输出 (B, num_classes, H, W)
        images: 原始图像批次 (B, 3, H, W)
        alpha: 透明度
    
    Returns:
        overlayed_images: 叠加后的图像列表
    """
    batch_size = outputs.shape[0]
    overlayed_images = []
    
    for i in range(batch_size):
        overlayed = process_segmentation_output(
            outputs[i:i+1], 
            images[i], 
            alpha=alpha
        )
        overlayed_images.append(overlayed)
    
    return overlayed_images
