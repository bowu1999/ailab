import os
import torch
import requests
import timm
import torch.nn as nn
from PIL import Image
from typing import Callable
from torchvision import transforms

from ailab import ailab_train
from ailab.metrics import Metric
from ailab.registry import MODELS, DATASETS, LOSSES, METRICS
from ailab import AnnotationFileLoadingDataset, load_dicts_from_jsonlines, std_transform

from example_config import cfg



_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.Lambda(lambda image: image.convert("RGB")),
    transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.5),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.3), ratio=(0.3, 3.3), value=0),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.3), ratio=(0.3, 3.3), value=0),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.3), ratio=(0.3, 3.3), value=0),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.3), ratio=(0.3, 3.3), value=0),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.3), ratio=(0.3, 3.3), value=0),
    transforms.Normalize(
        mean = [0.48145466, 0.4578275, 0.40821073],
        std = [0.26862954, 0.26130258, 0.27577711]
    )
])


@DATASETS.register_module()
class ClsRegImageDataset(AnnotationFileLoadingDataset):
    def __init__(
        self,
        annotation_file: str,
        x_key: str,
        cls_key: str,
        reg_key: str,
        dataset_type: str = "train",
    ):
        super().__init__(annotation_file = annotation_file, load_fun = load_dicts_from_jsonlines)
        self.x_key = x_key
        self.cls_key = cls_key
        self.reg_key = reg_key
        if dataset_type == "train":
            self.transform = _transform
        else:
            self.transform = std_transform

    def _get_sample(self, sample: dict) -> tuple:
        path = sample[self.x_key]
        cls = int(sample[self.cls_key])
        reg = float(sample[self.reg_key])
        image = Image.open(path)
        image = self.transform(image)

        return image.float(), dict(cls = cls, reg = reg)


@MODELS.register_module()
class MultiTaskResNet50(nn.Module):
    def __init__(self, num_classes=12, regression_output_size=1, pretrained_weights_path=None, freeze = False):
        super(MultiTaskResNet50, self).__init__()
        
        # 创建ResNet-50模型
        self.base_model = timm.create_model('resnet50', pretrained=False)
        
        if pretrained_weights_path is not None:
            if os.path.isfile(pretrained_weights_path):
                print(f"Loading weights from {pretrained_weights_path}")
                state_dict = torch.load(pretrained_weights_path)
                self.base_model.load_state_dict(state_dict)
            else:
                print(f"{pretrained_weights_path} does not exist or is not a file. Attempting to download...")
                
                # 确保目录存在
                os.makedirs(pretrained_weights_path, exist_ok=True)
                
                # 定义文件名
                default_pretrained_url = self.base_model.default_cfg['url']
                filename = os.path.basename(default_pretrained_url)
                save_path = os.path.join(pretrained_weights_path, filename)
                
                def download_file(url, save_path):
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        with open(save_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"Downloaded file to {save_path}")
                        return True
                    else:
                        print(f"Failed to download file. HTTP Status code: {response.status_code}")
                        return False
                
                downloaded = download_file(default_pretrained_url, save_path)
                if downloaded:
                    state_dict = torch.load(save_path)
                    self.base_model.load_state_dict(state_dict)
                else:
                    raise FileNotFoundError("Failed to download the pre-trained weights.")
        if freeze:
            # 先全部冻结
            for param in self.base_model.parameters():
                param.requires_grad = False
            # 选择解冻层
            for name, module in self.base_model.named_children():
                if name in ['layer3', 'layer4']:
                    for param in module.parameters():
                        param.requires_grad = True
        # 修改最后的全连接层
        num_features = self.base_model.fc.in_features
        self.classification_head = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.LogSoftmax(dim=1)
        )
        # 添加回归头
        self.regression_head = nn.Sequential(
            nn.Linear(num_features, regression_output_size)
        )
        # classification_head和regression_head默认是需要训练的，确保他们的参数：
        for param in self.classification_head.parameters():
            param.requires_grad = True

        for param in self.regression_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.base_model.forward_features(x)
        features = self.base_model.global_pool(features)
        features = torch.flatten(features, 1)
        
        class_output = self.classification_head(features)
        reg_output = self.regression_head(features)
        
        return dict(
            cls = class_output,
            reg = reg_output
        )


@LOSSES.register_module()
class MultiTaskLoss(nn.Module):
    def __init__(self, cls_weight=1.0, reg_weight=1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.MSELoss()

    def forward(self, outputs, targets):
        # 假设 outputs 是一个字典，包含 'cls' 和 'reg' 键
        # targets 也是一个字典，包含 'cls' 和 'reg' 键
        cls_loss = self.cls_loss_fn(outputs['cls'].squeeze(), targets['cls'])
        reg_loss = self.reg_loss_fn(outputs['reg'].squeeze(), targets['reg'])
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        return total_loss


@METRICS.register_module()
class ClassificationAccuracy(Metric):
    def __init__(self, topk=(1,)):
        self.topk = topk
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        # 假设分类头的输出在outputs['cls']中
        pred = outputs['cls']
        target = targets['cls']
        _, pred_topk = pred.topk(max(self.topk), 1, True, True)
        pred_topk = pred_topk.t()
        correct = pred_topk.eq(target.view(1, -1).expand_as(pred_topk))
        self.correct += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
        self.total += target.size(0)

    def compute(self):
        return self.correct / self.total * 100 if self.total > 0 else 0.0


@METRICS.register_module()
class RegressionMSE(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_squared_error = 0.0
        self.count = 0

    def update(self, outputs, targets):
        # 假设回归头的输出在outputs['reg']中
        pred = outputs['reg']
        target = targets['reg']
        self.sum_squared_error += torch.sum((pred - target) ** 2).item()
        self.count += target.numel()

    def compute(self):
        return self.sum_squared_error / self.count if self.count > 0 else 0.0


def main():
    ailab_train(cfg)

if __name__ == "__main__":
    main()