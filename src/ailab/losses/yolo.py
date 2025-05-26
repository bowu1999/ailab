import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou


class YOLOv5Loss(nn.Module):
    def __init__(self, anchors, strides, num_classes, hyp, debug=False):
        super().__init__()
        # ---- 省略原 init 中初始化锚点、超参等 ----
        anchors = torch.tensor(anchors, dtype=torch.float32)
        if anchors.ndimension() == 2 and anchors.shape[1] == 6:
            anchors = anchors.view(anchors.shape[0], -1, 2)
        assert anchors.ndimension() == 3 and anchors.shape[2] == 2
        self.anchors = anchors
        self.strides = torch.tensor(strides, dtype=torch.float32)
        self.nc = num_classes
        self.nl = anchors.shape[0]
        self.na = anchors.shape[1]
        self.hyp = hyp
        self.balance = [4.0,1.0,0.4] if self.nl==3 else [1.0]*self.nl
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp.get("cls_pw",1.0)]))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp.get("obj_pw",1.0)]))
        # debug flag
        self.debug = debug
        self._debug_done = False

    def forward(self, preds, targets):
        device = preds[0].device
        # 第一次进来打一下 targets 原样
        if self.debug and not self._debug_done:
            print(">>> [Loss] raw targets list:")
            for i, t in enumerate(targets):
                print(f"  img {i}: shape={t.shape}, values=\n{t}")
        # ensure on device
        targets = [t.to(device) for t in targets]
        # build targets
        tcls, tbox, indices, anch = self.build_targets(preds, targets, device)
        # loss 初始化
        loss = torch.zeros(3, device=device)

        # 打印 build_targets 的 key 信息（仅一次）
        if self.debug and not self._debug_done:
            print(">>> [Loss] after build_targets:")
            for lvl, (cls_i, box_i, idx_i) in enumerate(zip(tcls, tbox, indices)):
                print(f"  level {lvl}: matched instances = {cls_i.shape[0]}")
            self._debug_done = True

        # 原有的 loss 计算逻辑（略微格式化）
        for i, pi in enumerate(preds):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[...,0], device=device)
            n = b.shape[0]
            if n:
                # box loss
                pxy = pi[b,a,gj,gi,0:2].sigmoid()
                pwh = pi[b,a,gj,gi,2:4].sigmoid()
                pbox = torch.cat((pxy, pwh),1)
                giou = self.bbox_giou(pbox, tbox[i])
                loss[0] += (1.0-giou).mean()
                # obj loss 标签
                tobj[b,a,gj,gi] = giou.detach().clamp(0)
                # cls loss
                if self.nc > 1:
                    pred_cls = pi[b,a,gj,gi,5:]
                    t = torch.zeros_like(pred_cls, device=device)
                    tcls_i = tcls[i].to(device)
                    idx = torch.arange(n, device=device)
                    t[idx, tcls_i] = 1.0
                    loss[2] += self.BCEcls(pred_cls, t)
            # obj loss
            loss[1] += self.BCEobj(pi[...,4], tobj) * self.balance[i]

        # 加权
        loss[0] *= self.hyp['box']
        loss[1] *= self.hyp['obj']
        loss[2] *= self.hyp['cls']
        return loss.sum()

    def build_targets(self, preds, targets, device):
        na, nl = self.na, self.nl
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=device)

        # 拼接所有 targets
        targets_full = []
        for i, t in enumerate(targets):
            if t.numel():
                img_idx = torch.full((t.size(0),1), i, device=device, dtype=t.dtype)
                targets_full.append(torch.cat([img_idx, t],1))
        if targets_full:
            targets_tensor = torch.cat(targets_full,0)
        else:
            targets_tensor = torch.zeros((0,6), device=device)

        # 再次 debug：打印 targets_tensor
        if self.debug and not self._debug_done:
            print(f">>> [build_targets] targets_tensor shape={targets_tensor.shape}")
            print(targets_tensor)

        # 为每个尺度分配
        for i in range(nl):
            anchors_i = self.anchors[i].to(device)
            shape = preds[i].shape
            gain[2:6] = torch.tensor(shape)[[2,3,2,3]].to(device)
            if targets_tensor.numel():
                # repeat & concat anchor idx
                t = targets_tensor.repeat(na,1,1)
                a = torch.arange(na,device=device).view(na,1).repeat(1,targets_tensor.size(0)).view(-1)
                tt = torch.cat([t.view(-1,6), a.unsqueeze(1).float()],1)
                # scale to grid
                tt[:,2:6] *= gain[2:6]
                # ratio with anchors
                r = tt[:,4:6] / anchors_i[tt[:,6].long()]
                r = torch.max(r, 1./r).max(1)[0]
                # debug mask stats
                if self.debug and not self._debug_done:
                    print(f">>> [build_targets][level {i}] before mask {tt.size(0)}, r.min={r.min():.3f}, r.max={r.max():.3f}")
                mask = r < 4.0
                tt = tt[mask]
                if self.debug and not self._debug_done:
                    print(f">>> [build_targets][level {i}] after  mask {tt.size(0)}")
                if tt.size(0):
                    b = tt[:,0].long(); c = tt[:,1].long()
                    gxy = tt[:,2:4]; gwh = tt[:,4:6]
                    gij = gxy.long(); gi, gj = gij.t()
                    indices.append((b, tt[:,6].long(), gj, gi))
                    tbox.append(torch.cat((gxy-gij, gwh),1))
                    anch.append(anchors_i[tt[:,6].long()])
                    tcls.append(c)
                    continue
            # no targets for this level
            indices.append((torch.zeros(0,device=device,dtype=torch.long),)*4)
            tbox.append(torch.zeros((0,4),device=device))
            anch.append(torch.zeros((0,2),device=device))
            tcls.append(torch.zeros((0,),device=device,dtype=torch.long))

        return tcls, tbox, indices, anch

    @staticmethod
    def bbox_giou(pbox, tbox):
        # unchanged
        """
        pbox, tbox: (n,4)  格式都是 (cx,cy,w,h)
        返回每对框的 GIoU diagonal
        """
        # 转成 x1,y1,x2,y2
        px = pbox[:, :2]
        pw = pbox[:, 2:]
        p1 = px - pw / 2
        p2 = px + pw / 2
        p_rect = torch.cat([p1, p2], 1)

        tx = tbox[:, :2]
        tw = tbox[:, 2:]
        t1 = tx - tw / 2
        t2 = tx + tw / 2
        t_rect = torch.cat([t1, t2], 1)

        return generalized_box_iou(p_rect, t_rect).diagonal()
