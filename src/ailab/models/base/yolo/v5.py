import math
import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou, nms


# --------------------------------------------
# Helpers
# --------------------------------------------
def autopad(k, p=None):
    # Pad to 'same'
    return p if p is not None else k // 2


# --------------------------------------------
# Basic Modules
# --------------------------------------------
class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU/Identity"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    """Focus wh information into channels"""
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p)

    def forward(self, x):
        # [b, c, h, w] -> [b, 4c, h/2, w/2]
        return self.conv(torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        ], dim=1))


class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """C3: CSP Bottleneck with 3 conv"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class SPPF(nn.Module):
    """SPP-Fast"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class Concat(nn.Module):
    """Concatenate along channel dim"""
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


# --------------------------------------------
# Detection Head
# --------------------------------------------
class Detect(nn.Module):
    """Detection head for 3 feature maps"""
    def __init__(self, nc=80, anchors=(), ch=(), strides=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.na = len(anchors[0]) // 2
        self.nl = len(anchors)
        
        # 注册为buffer
        self.register_buffer('anchors', torch.tensor(anchors, dtype=torch.float32).view(self.nl, -1, 2))
        self.register_buffer('stride', torch.tensor(strides, dtype=torch.float32))
        
        self.grid = [torch.zeros(1)] * self.nl
        self.m = nn.ModuleList(
            nn.Conv2d(ch[i], self.no * self.na, 1) for i in range(self.nl)
        )
        
        # 初始化偏置
        self._initialize_biases()

    def _initialize_biases(self):
        """初始化检测层的偏置"""
        import math
        
        for mi, s in zip(self.m, self.stride):
            # 获取bias的shape
            b = mi.bias.data.view(self.na, -1)
            
            # 创建新的bias值
            new_bias = torch.zeros_like(b)
            
            # 复制原始值
            new_bias.copy_(b)
            
            # Objectness bias (使预测初始值约为0.01)
            new_bias[:, 4] += math.log(0.01 / 0.99)
            
            # Class bias (使预测初始值约为1/nc)
            if self.nc > 1:
                new_bias[:, 5:] += math.log(0.001 / (self.nc - 0.999))
            
            # 设置新的bias
            mi.bias.data.copy_(new_bias.view(-1))

    def forward(self, features):
        raw_preds = []
        decoded = []
        
        for i, x in enumerate(features):
            bs, _, ny, nx = x.shape
            p = self.m[i](x).view(bs, self.na, self.no, ny, nx)
            p = p.permute(0, 1, 3, 4, 2).contiguous()
            raw_preds.append(p)
            
            # 如果grid不匹配就重新生成
            if self.grid[i].shape[2:] != (ny, nx):
                yv, xv = torch.meshgrid(
                    torch.arange(ny, device=p.device),
                    torch.arange(nx, device=p.device),
                    indexing='ij'
                )
                self.grid[i] = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()
            
            # 解码过程，仅用于推理
            if not self.training:
                psig = p.sigmoid()
                xy = (psig[..., :2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                wh = (psig[..., 2:4] * 2) ** 2 * self.anchors[i].view(1, self.na, 1, 1, 2)
                conf = psig[..., 4:5]
                cls = psig[..., 5:]
                decoded.append(torch.cat([xy, wh, conf, cls], dim=-1))
        
        return raw_preds if self.training else decoded

# --------------------------------------------
# YOLOv5 Model
# --------------------------------------------
class YOLOv5(nn.Module):
    def __init__(
        self,
        nc = 80,
        anchors = (),
        ch = (64, 128, 256, 512, 1024),
        strides = (8, 16, 32),
        conf_thres = 0.25,
        iou_thres = 0.45
    ):
        super().__init__()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # Stem
        self.stem = Focus(3, ch[0], k=3, s=1)
        # Backbone
        self.backbone = nn.Sequential(
            Conv(ch[0], ch[1], 3, 2), C3(ch[1], ch[1], n=3),
            Conv(ch[1], ch[2], 3, 2), C3(ch[2], ch[2], n=9),
            Conv(ch[2], ch[3], 3, 2), C3(ch[3], ch[3], n=9),
            Conv(ch[3], ch[4], 3, 2), SPPF(ch[4], ch[4], k=5)
        )
        # PANet
        self.conv_P5 = Conv(ch[4], ch[3], 1, 1)
        self.conv_P4 = Conv(ch[3], ch[2], 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat = Concat(1)
        self.c3_P5 = C3(ch[3]*2, ch[3], n=3, shortcut=False)
        self.c3_P4 = C3(ch[2]*2, ch[2], n=3, shortcut=False)
        self.down1 = Conv(ch[2], ch[2], 3, 2)
        self.down2 = Conv(ch[3], ch[3], 3, 2)
        self.c3_N3 = C3(ch[2]+ch[3], ch[3], n=3, shortcut=False)
        self.c3_N4 = C3(ch[3]+ch[4], ch[4], n=3, shortcut=False)
        # Head
        self.detect = Detect(nc, anchors, ch=[ch[2], ch[3], ch[4]], strides=strides)

    def forward(self, x):
        # Stem + Backbone
        x = self.stem(x)
        x = self.backbone[0](x); x = self.backbone[1](x)
        x = self.backbone[2](x); x = self.backbone[3](x); P3 = x
        x = self.backbone[4](x); x = self.backbone[5](x); P4 = x
        x = self.backbone[6](x); x = self.backbone[7](x); P5 = x
        # PANet Top-Down
        P5u = self.upsample(self.conv_P5(P5))
        P4 = self.c3_P5(self.concat([P5u, P4]))
        P4u = self.upsample(self.conv_P4(P4))
        P3 = self.c3_P4(self.concat([P4u, P3]))
        # PANet Bottom-Up
        N3 = self.c3_N3(self.concat([self.down1(P3), P4]))
        N4 = self.c3_N4(self.concat([self.down2(N3), P5]))
        # Detect
        out = self.detect([P3, N3, N4])
        # 训练模式直接返回 raw/decoded，由 Detect 控制
        if self.training:
            return out
        # 推理模式：对 decoded 输出做 NMS 合并三层
         # 1) 将三个尺度展平成 (bs, num_preds, no)
        preds = []
        for p in out:
            bs, na, ny, nx, no = p.shape
            preds.append(p.view(bs, -1, no))
        preds = torch.cat(preds, dim=1)  # (bs, N, 5+nc)

        results = []
        for pi in preds:  # pi: (N, 5+nc)
            # 2) 取出 xywh、conf、cls_prob
            xy = pi[:, :2]
            wh = pi[:, 2:4]
            conf = pi[:, 4]               # already sigmoid 过
            cls_p = pi[:, 5:]             # already sigmoid 过

            # 3) 从 [cx,cy,w,h] 转到 [x1,y1,x2,y2]
            x1 = xy[:, 0] - wh[:, 0] / 2
            y1 = xy[:, 1] - wh[:, 1] / 2
            x2 = xy[:, 0] + wh[:, 0] / 2
            y2 = xy[:, 1] + wh[:, 1] / 2
            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # 4) 组合得分并过滤
            scores, labels = (conf.unsqueeze(1) * cls_p).max(1)
            mask = scores > self.conf_thres
            boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

            # 5) NMS
            if boxes.numel():
                keep = nms(boxes, scores, self.iou_thres)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            results.append((boxes, scores, labels))

        return results, out
