import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from pathlib import Path
from typing import List, Union
import torchvision.transforms as T
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from ._base import FileAnnotationDataset



# --------------------------------------------
# Dataset: Pascal VOC-style
# --------------------------------------------
class VOCDataset(Dataset):
    """Pascal VOC format dataset loader"""
    def __init__(self, root, img_size=640, transform=None):
        self.root = root
        self.img_dir = os.path.join(root, 'JPEGImages')
        self.label_dir = os.path.join(root, 'Annotations')
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir)]
        self.img_size = img_size
        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ColorJitter(0.2,0.2,0.2,0.1),
            T.ToTensor(),
        ])
        self.class_map = {c:i for i,c in enumerate(
            ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
             'chair','cow','diningtable','dog','horse','motorbike','person',
             'pottedplant','sheep','sofa','train','tvmonitor'])}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        xml_path = os.path.join(self.label_dir, img_id + '.xml')
        img = torchvision.io.read_image(img_path).float() / 255.0
        img = T.Resize((self.img_size, self.img_size))(img)
        h0, w0 = img.shape[1], img.shape[2]
        # parse annotations
        tree = ET.parse(xml_path)
        root = tree.getroot()
        targets = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            cls_id = self.class_map[cls]
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            # normalize
            x1n, y1n = x1/w0, y1/h0
            x2n, y2n = x2/w0, y2/h0
            targets.append([cls_id, x1n, y1n, x2n, y2n])
        targets = torch.tensor(targets) if targets else torch.zeros((0,5))
        return img, targets



class YoloFileDataset(FileAnnotationDataset): # TOUPDATE
    """
    YOLO dataset where each raw sample is a dict or list specifying
    image path and bounding boxes.
    Expected raw format: {
      'img_path': str or Path,
      'boxes': List[List[float]]  # each [class_id, x_center, y_center, w, h] normalized
    }
    """
    def __init__(
        self,
        annotation_file: Union[str, Path, List[str]],
        load_fun,
        image_key: str = "image",
        bbox_key: str = "bbox",
        img_size: int = 640,
        transforms = None
    ):
        super().__init__(annotation_file, load_fun)
        self.image_key = image_key
        self.bbox_key = bbox_key
        self.img_size = img_size
        self.transforms = transforms

    def _get_sample(self, raw):
        img_path = Path(raw[self.image_key])
        boxes = raw[self.bbox_key]
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        
        return dict(image = img, bbox = boxes)
