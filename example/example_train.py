import torch
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio

from ailab import ailab_train
from ailab.losses import LabBaseLoss
from ailab.metrics import LabBaseMetric
from ailab.registry import METRICS, LOSSES

from config import cfg


@LOSSES.register_module()
class ReconstructionLoss(LabBaseLoss):
    """
    基于 BaseLoss 的重建损失：
      - forward(self, pred, mask, target) 完全声明需要的字段
      - 框架通过 call_fn 自动绑定 data['pred'], data['mask'], data['target']
    """
    def __init__(self):
        # BaseLoss 会保存 cfg，可在此添加更多超参
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
          pred   (B, N, C): 模型输出的 patch 重建
          mask   (B, N)   : mask 掩码，True 表示该 patch 丢弃，需要重建
          target (B, N, C): 原始图像 patch 特征
        Returns:
          单标量损失，只在 mask=True 的位置计算 MSE
        """
        # 1) 展平到 [B*N, C] 与 [B*N]
        B, N, C = pred.shape
        pred_flat   = pred.reshape(B * N, C)
        target_flat = target.reshape(B * N, C)
        mask_flat   = mask.reshape(B * N)

        # 2) 只在被掩码的部分计算
        pred_sel   = pred_flat[mask_flat]
        target_sel = target_flat[mask_flat]

        # 3) 返回 MSELoss
        return self.criterion(pred_sel, target_sel)




@METRICS.register_module()
class MyPeakSignalNoiseRatio(LabBaseMetric, PeakSignalNoiseRatio):
    """
    基于 torchmetrics.PSNR 的封装：
      - update(self, preds, targets) 中处理 preds/targets
      - compute() 调用父类 PeakSignalNoiseRatio.compute
    """
    def __init__(self, data_range: float = 1.0, **cfg):
        # LabBaseMetric 构造器会保存 cfg，并注册 reset/update/compute 的规范接口
        LabBaseMetric.__init__(self, **cfg)
        # 调用 torchmetrics 中的 PSNR 初始化
        PeakSignalNoiseRatio.__init__(self, data_range=data_range)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        :param preds:   重建图像 [B, C, H, W]
        :param targets: 原始图像 [B, C, H, W]
        """
        # 直接调用 torchmetrics 的 update
        super(PeakSignalNoiseRatio, self).update(preds, targets)

    def compute(self) -> torch.Tensor:
        """
        必须实现抽象方法 compute()，直接调用 torchmetrics 的 compute 即可
        """
        return PeakSignalNoiseRatio.compute(self)


if __name__ == "__main__":
    ailab_train(cfg)