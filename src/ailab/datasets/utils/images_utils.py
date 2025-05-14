import cv2
import matplotlib.pyplot as plt


def draw_bounding_boxes(image, bboxes, bbox_format='xyxy', output_path=None):
    """
    在给定的图片上使用bboxes数据绘制边界框。
    
    参数:
    - image: 原始图片
    - bboxes: 边界框列表，每个元素可以是 (x_min, y_min, x_max, y_max) 或者 (label, confidence, x_min, y_min, x_max, y_max)
              或者 (x, y, width, height) 或者 (label, confidence, x, y, width, height)
    - bbox_format: 'xyxy' 表示左上角和右下角坐标；'xywh' 表示左上角坐标、宽度和高度
    - output_path: 保存带有标注框的新图的路径（可选）
    """
    for bbox in bboxes:
        if bbox_format == 'xyxy':
            if len(bbox) == 4:  # 只有框信息
                x_min, y_min, x_max, y_max = bbox
                label = None
                confidence = None
            elif len(bbox) == 6:  # 包含标签和置信度
                label, confidence, x_min, y_min, x_max, y_max = bbox
            else:
                raise ValueError("每个边界框应包含4个或6个值")
        elif bbox_format == 'xywh':
            if len(bbox) == 4:  # 只有框信息
                x, y, width, height = bbox
                x_min, y_min, x_max, y_max = x, y, x + width, y + height
                label = None
                confidence = None
            elif len(bbox) == 6:  # 包含标签和置信度
                label, confidence, x, y, width, height = bbox
                x_min, y_min, x_max, y_max = x, y, x + width, y + height
            else:
                raise ValueError("每个边界框应包含4个或6个值")
        else:
            raise ValueError("bbox_format 应为 'xyxy' 或 'xywh'")
        
        # 绘制矩形框
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 255, 0), thickness=2)
        
        # 如果提供了标签和置信度，则显示它们
        if label is not None and confidence is not None:
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    
    # 如果提供了输出路径，则保存图像
    if output_path is not None:
        cv2.imwrite(output_path, image)
    else:
        return image


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