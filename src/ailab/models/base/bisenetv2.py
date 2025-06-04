import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """基础卷积块：Conv + BN + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class StemBlock(nn.Module):
    """BiSeNetV2的Stem块，用于初始特征提取"""
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 16, 3, 2, 1)  # 1/2
        self.branch1 = nn.Sequential(
            ConvBNReLU(16, 16, 1, 1, 0),
            ConvBNReLU(16, 32, 3, 2, 1)  # 1/4
        )
        self.branch2 = nn.MaxPool2d(3, 2, 1)  # 1/4
        self.combine = ConvBNReLU(48, 32, 3, 1, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], dim=1)
        return self.combine(x)

class GatherExpansion(nn.Module):
    """Gather-and-Expansion Layer"""
    def __init__(self, in_channels, out_channels, stride=1, e=6):
        super().__init__()
        mid_channels = in_channels * e
        self.conv1 = ConvBNReLU(in_channels, in_channels, 3, 1, 1)
        
        if stride == 2:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.use_shortcut = stride == 1 and in_channels == out_channels
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        if self.use_shortcut:
            x = x + identity
        return F.relu(x)

class DetailBranch(nn.Module):
    """细节分支：保留空间细节信息"""
    def __init__(self):
        super().__init__()
        # Stage 1: 1/2
        self.s1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, 2, 1),
            ConvBNReLU(64, 64, 3, 1, 1)
        )
        # Stage 2: 1/4
        self.s2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, 2, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
            ConvBNReLU(64, 64, 3, 1, 1)
        )
        # Stage 3: 1/8 - 修正输出通道数为128
        self.s3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1)
        )
    
    def forward(self, x):
        x1 = self.s1(x)  # 1/2, 64 channels
        x2 = self.s2(x1)  # 1/4, 64 channels
        x3 = self.s3(x2)  # 1/8, 128 channels
        return x3

class SemanticBranch(nn.Module):
    """语义分支：提取高级语义信息"""
    def __init__(self):
        super().__init__()
        self.stem = StemBlock()
        # Stage 3: 1/8 - 修正输出通道数为128
        self.s3 = nn.Sequential(
            GatherExpansion(32, 32),
            GatherExpansion(32, 128, 2)  # 改为输出128通道
        )
        # Stage 4: 1/16
        self.s4 = nn.Sequential(
            GatherExpansion(128, 128),  # 调整输入通道
            GatherExpansion(128, 128, 2)
        )
        # Stage 5: 1/32
        self.s5 = nn.Sequential(
            GatherExpansion(128, 128),
            GatherExpansion(128, 128),
            GatherExpansion(128, 128),
            GatherExpansion(128, 128, 2)
        )
    
    def forward(self, x):
        x = self.stem(x)  # 1/4, 32 channels
        x3 = self.s3(x)   # 1/8, 128 channels
        x4 = self.s4(x3)  # 1/16, 128 channels
        x5 = self.s5(x4)  # 1/32, 128 channels
        return x3, x4, x5

class BilateralGuidedAggregation(nn.Module):
    """双边引导聚合层"""
    def __init__(self, channels=128):
        super().__init__()
        # 细节分支处理
        self.detail_conv = nn.Sequential(
            ConvBNReLU(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        )
        # 语义分支处理
        self.semantic_conv = nn.Sequential(
            ConvBNReLU(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        )
        # 聚合
        self.conv = ConvBNReLU(channels, channels, 3, 1, 1)
    
    def forward(self, detail, semantic):
        # 上采样语义特征到细节特征的尺寸
        semantic = F.interpolate(semantic, size=detail.shape[2:], mode='bilinear', align_corners=True)
        
        # 双边引导
        detail_att = torch.sigmoid(self.detail_conv(detail))
        semantic_att = torch.sigmoid(self.semantic_conv(semantic))
        
        detail = detail * semantic_att
        semantic = semantic * detail_att
        
        # 聚合
        x = detail + semantic
        return self.conv(x)

class SegmentHead(nn.Module):
    """分割头"""
    def __init__(self, in_channels, inter_channels, num_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, num_classes, 1, 1, 0)
    
    def forward(self, x, size=None):
        x = self.conv(x)
        x = self.conv_out(x)
        if size is not None:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x


class BiSeNetV2(nn.Module):
    """BiSeNetV2主模型"""
    def __init__(self, num_classes=19):  # Cityscapes有19个类别
        super().__init__()
        # 两个分支
        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        
        # 双边引导聚合
        self.bga = BilateralGuidedAggregation(128)
        
        # 分割头
        self.head = SegmentHead(128, 128, num_classes)
        
        # 辅助头（训练时使用）
        self.aux_heads = nn.ModuleList([
            SegmentHead(128, 128, num_classes),  # s3 - 修正输入通道
            SegmentHead(128, 128, num_classes),  # s4
            SegmentHead(128, 128, num_classes)   # s5
        ])
    
    def forward(self, x):
        # 获取输入尺寸
        size = x.shape[2:]
        
        # 两个分支的前向传播
        detail = self.detail_branch(x)  # 128 channels
        s3, s4, s5 = self.semantic_branch(x)  # all 128 channels
        
        # 双边引导聚合
        x = self.bga(detail, s3)
        
        # 主输出
        out = self.head(x, size)
        
        if self.training:
            # 训练时返回多个输出用于深监督
            aux_outs = [
                self.aux_heads[0](s3, size),
                self.aux_heads[1](s4, size),
                self.aux_heads[2](s5, size)
            ]
            return out, aux_outs
        else:
            return out
