import torch
import torch.nn as nn


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
        # 关键修改：把这两个改为 buffer
        self.register_buffer('anchors',  torch.tensor(anchors, dtype=torch.float32).view(self.nl, -1, 2))
        self.register_buffer('stride',  torch.tensor(strides, dtype=torch.float32))
        self.grid = [torch.zeros(1)] * self.nl
        self.m = nn.ModuleList(
            nn.Conv2d(ch[i], self.no * self.na, 1) for i in range(self.nl)
        )

    def forward(self, features):
        raw_preds = []
        decoded = []
        for i, x in enumerate(features):
            bs, _, ny, nx = x.shape
            p = self.m[i](x).view(bs, self.na, self.no, ny, nx)
            p = p.permute(0, 1, 3, 4, 2).contiguous()  # 原始输出 logits
            raw_preds.append(p)
            # 如果 grid 不匹配就重新生成
            if self.grid[i].shape[2:] != (ny, nx):
                yv, xv = torch.meshgrid(
                    torch.arange(ny, device=p.device),
                    torch.arange(nx, device=p.device),
                    indexing = 'ij'
                )
                self.grid[i] = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float().to(p.device)
            # 解码过程，仅用于推理
            psig = p.sigmoid().clone()
            xy = (psig[..., :2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
            wh = (psig[..., 2:4] * 2) ** 2 * self.anchors[i].view(1, self.na, 1, 1, 2)
            conf = psig[..., 4:5]
            cls = psig[..., 5:]
            decoded.append(torch.cat([xy, wh, conf, cls], dim=-1))

        # 如果是训练阶段，返回 raw logits；如果是推理阶段，返回解码结果
        # 长度 3 的列表，表示每个层的输出
        # [(batch_size, na, ny, nx, no), (batch_size, na, ny, nx, no), (batch_size, na, ny, nx, no)]
        return raw_preds if self.training else decoded


# --------------------------------------------
# YOLOv5 Model
# --------------------------------------------
class YOLOv5(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(64,128,256,512,1024), strides=(8,16,32), conf_thres=0.25, iou_thres=0.45):
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
        self.concat   = Concat(1)
        self.c3_P5    = C3(ch[3]*2, ch[3], n=3, shortcut=False)
        self.c3_P4    = C3(ch[2]*2, ch[2], n=3, shortcut=False)
        self.down1    = Conv(ch[2], ch[2], 3, 2)
        self.down2    = Conv(ch[3], ch[3], 3, 2)
        self.c3_N3    = C3(ch[2]+ch[3], ch[3], n=3, shortcut=False)
        self.c3_N4    = C3(ch[3]+ch[4], ch[4], n=3, shortcut=False)
        # Head
        self.detect = Detect(nc, anchors, ch=[ch[2],ch[3],ch[4]], strides=strides)

    def forward(self, x):
        # Stem + Backbone
        x = self.stem(x)
        x = self.backbone[0](x); x = self.backbone[1](x)
        x = self.backbone[2](x); x = self.backbone[3](x); P3 = x
        x = self.backbone[4](x); x = self.backbone[5](x); P4 = x
        x = self.backbone[6](x); x = self.backbone[7](x); P5 = x
        # PANet Top-Down
        P5u = self.upsample(self.conv_P5(P5))
        P4  = self.c3_P5(self.concat([P5u, P4]))
        P4u = self.upsample(self.conv_P4(P4))
        P3  = self.c3_P4(self.concat([P4u, P3]))
        # PANet Bottom-Up
        N3  = self.c3_N3(self.concat([self.down1(P3), P4]))
        N4  = self.c3_N4(self.concat([self.down2(N3), P5]))
        # Detect
        out = self.detect([P3, N3, N4])
        # 训练模式直接返回 raw/decoded，由 Detect 控制
        if self.training:
            return out
        # 推理模式：对 decoded 输出做 NMS 合并三层
        from torchvision.ops import nms
        preds = []
        for p in out:
            bs, na, ny, nx, no = p.shape
            preds.append(p.view(bs, -1, no))
        preds = torch.cat(preds, dim=1)  # (bs, num_pred, 5nc)

        results = []
        for xi in preds:  # 逐图处理
            # 取框和得分
            boxes  = xi[..., :4]                    # x1,y1,x2,y2
            obj    = xi[..., 4:5]                   # objectness
            cls_p  = xi[..., 5:]                    # class probs
            scores, labels = (obj * cls_p).max(1)   # 联合得分与类别

            # 过滤低置信
            mask = scores > self.conf_thres
            boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

            # NMS
            keep = nms(boxes, scores, self.iou_thres)
            results.append((boxes[keep], scores[keep], labels[keep]))

        return results


if __name__ == '__main__':
    anchors = [
        [10,13, 16,30, 33,23],
        [30,61, 62,45, 59,119],
        [116,90, 156,198, 373,326],
    ]
    model = YOLOv5(nc=80, anchors=anchors)
    x = torch.randn(1,3,640,640)
    outs = model(x)
    for o in outs:
        print(o.shape)
