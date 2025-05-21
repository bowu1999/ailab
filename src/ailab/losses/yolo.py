import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou

from ._base import LabBaseLoss


class YOLONormalizedLoss(LabBaseLoss):
    """
    改进版 YOLO 损失，支持多尺度 anchor 匹配 + CIoU
    输入：
        anchors: list of per-scale anchor lists, e.g.
                 [[10,13,16,30,33,23], [30,61,62,45,59,119], ...]
        strides: list/tuple of strides 对应每个尺度, e.g. (8,16,32)
        nc:       类别数
        lambda_box, lambda_obj, lambda_cls: 各损失项权重
        iou_threshold: 负样本 IoU 忽略阈值
        use_ciou: 是否启用 CIoU
    """
    def __init__(
        self,
        anchors,
        strides,
        nc = 80,
        lambda_box = 5.0,
        lambda_obj = 1.0,
        lambda_cls = 1.0,
        iou_threshold = 0.5,
        use_ciou = True
    ):
        super().__init__(
            nc = nc,
            lambda_box=lambda_box,
            lambda_obj=lambda_obj,
            lambda_cls=lambda_cls
        )
        self.nc = nc
        self.lb = lambda_box
        self.lo = lambda_obj
        self.lc = lambda_cls
        self.use_ciou = use_ciou
        self.iou_threshold = iou_threshold
        # (nl, na, 2)
        self.register_buffer('anchors', torch.tensor(anchors, dtype=torch.float32).view(len(anchors), -1, 2))
        self.register_buffer('strides', torch.tensor(strides, dtype=torch.float32))
        # 损失函数
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    @staticmethod
    def _sigmoid(x):
        return x.sigmoid()

    def decode_boxes(self, raw, grid, stride, anchor_wh):
        """
        将网络输出 raw(tx,ty,tw,th) 解码为 (x1,y1,x2,y2)
        参数:
            raw:        (M,4) tensor
            grid:       (M,2) tensor: 每个预测对应的 (gi, gj)
            stride:     标量
            anchor_wh:  (M,2) tensor：每个预测对应的 anchor 宽高

        返回: (M,4) x1,y1,x2,y2
        """
        tx, ty, tw, th = raw.unbind(1)
        gi, gj = grid.unbind(1)
        # 中心坐标
        cx = (self._sigmoid(tx)*2 - 0.5 + gi) * stride
        cy = (self._sigmoid(ty)*2 - 0.5 + gj) * stride
        # 宽高
        aw, ah = anchor_wh.unbind(1)
        w = (self._sigmoid(tw)*2)**2 * aw
        h = (self._sigmoid(th)*2)**2 * ah
        # 转为 x1,y1,x2,y2
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return torch.stack([x1, y1, x2, y2], dim=1)

    def forward(self, pred, target):
        """
        参数:
            pred: list of tensors, 每个尺度 (bs, na, ny, nx, 5+nc)
            target: list of dict 或 list of tensor(bs, nt,6)
                    若为 dict，则包含 'boxes' (Ni,4) 与 'labels' (Ni,)
        返回:
            scalar loss
        """
        device = pred[0].device
        # torch.Size([4, 3, 80, 80, 85])
        bs = pred[0].size(0)
        # 如果 target 是 list of dict，则转换
        if isinstance(target, (list, tuple)) and isinstance(target[0], dict):
            targets = []
            for b in range(bs):
                boxes = target[b]['boxes']    # (Ni,4), [cx,cy,w,h]
                labels = target[b]['labels']  # (Ni,)
                Ni = labels.size(0)
                if Ni == 0:
                    targets.append(torch.zeros((0,6), device=device))
                    continue
                idx = torch.full((Ni,1), b, device=device, dtype=torch.long)
                cls = labels.unsqueeze(1).to(device)
                tgt = torch.cat([idx.float(), cls.float(), boxes], dim=1)
                targets.append(tgt)
        else:
            targets = target  # 假设已经是 list of tensor(bs,nt,5)
        l_box = torch.zeros(1, device=device)
        l_obj = torch.zeros(1, device=device)
        l_cls = torch.zeros(1, device=device)
        # 遍历 batch 中每张图
        for b in range(bs):
            tb = targets[b]  # (nt,6)
            if tb.numel() == 0:
                # 全负样本
                for pi in pred:
                    l_obj += self.bce_loss(pi[b, ..., 4], torch.zeros_like(pi[b, ..., 4]))
                continue
            # 逐尺度处理
            for i, pi in enumerate(pred):
                stride = self.strides[i]
                anchors_i = self.anchors[i]  # (na,2)
                bs_i, na, ny, nx, no = pi.shape
                # 平铺预测
                p = pi[b].view(na*ny*nx, no)  # (na*ny*nx, 5+nc)
                # 网格索引
                gy, gx = torch.meshgrid(
                    torch.arange(ny, device=device),
                    torch.arange(nx, device=device),
                    indexing='ij'
                )
                grid = torch.stack((gx, gy), dim=2).view(-1,2).float()  # (ny*nx,2)
                grid = grid.repeat(na, 1)                              # (na*ny*nx,2)
                anchor_wh = anchors_i.repeat(ny*nx, 1)                 # (na*ny*nx,2)
                # --- 1) 置信度负样本损失 ---
                obj_target = torch.zeros((na*ny*nx,), device=device)
                l_obj += self.bce_loss(p[:,4], obj_target)
                # --- 2) 正样本匹配与三部分损失 ---
                for gt in tb:
                    cls, gx_n, gy_n, gw_n, gh_n = gt
                    # 计算所在 cell
                    gi = int(gx_n * nx)
                    gj = int(gy_n * ny)
                    # 匹配最佳 anchor
                    rel_wh = torch.tensor([gw_n * nx, gh_n * ny], device=device)
                    ratios = (rel_wh / anchors_i).log().abs().sum(1)
                    best_n = torch.argmin(ratios)
                    # flat 索引
                    idx = best_n * (ny*nx) + gj * nx + gi
                    # 构建回归 target(tx,ty,tw,th)
                    tx = gx_n*nx - gi
                    ty = gy_n*ny - gj
                    tw = torch.log((gw_n*nx) / anchors_i[best_n,0] + 1e-16)
                    th = torch.log((gh_n*ny) / anchors_i[best_n,1] + 1e-16)
                    target_raw = torch.stack([tx, ty, tw, th], dim=0).to(device)
                    # 坐标损失
                    if self.use_ciou:
                        # 解码后计算 CIoU
                        pred_raw = p[idx, 0:4].unsqueeze(0)
                        true_raw = target_raw.unsqueeze(0)
                        # 解码
                        pred_box = self.decode_boxes(pred_raw, grid[idx:idx+1], stride, anchor_wh[idx:idx+1])
                        true_box = self.decode_boxes(true_raw, grid[idx:idx+1], stride, anchor_wh[idx:idx+1])
                        giou = generalized_box_iou(pred_box, true_box).diag().clamp(-1+1e-7, 1-1e-7)
                        l_box += (1 - giou).sum()
                    else:
                        l_box += self.mse_loss(p[idx, 0:4], target_raw)
                    # 置信度正样本
                    obj_target[idx] = 1.0
                    l_obj += self.bce_loss(p[idx,4], torch.tensor(1.0, device=device))
                    # 分类损失
                    cls_target = torch.zeros((self.nc,), device=device)
                    cls_target[int(cls)] = 1.0
                    l_cls += self.bce_loss(p[idx,5:], cls_target)

        # 加权并归一
        loss = (self.lb * l_box + self.lo * l_obj + self.lc * l_cls) / bs
        return loss
