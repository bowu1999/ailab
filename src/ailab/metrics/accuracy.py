import torch
from ._base import LabBaseMetric


class Accuracy(LabBaseMetric):
    """
    Top‑K 分类准确率
    子类只需实现 reset/update/compute，字段名由配置中的 mapping 决定
    """
    def __init__(self, topk: int = 1, **cfg):
        super().__init__(**cfg)
        self.topk = topk
        self.reset()

    def reset(self):
        # 清空累积状态
        self.correct = 0
        self.total = 0

    def update(self, pred: torch.Tensor, label: torch.Tensor):
        """
        Args:
            pred: [B, C] 的预测 logits 或概率
            label: [B] 的真实类别标签
        """
        with torch.no_grad():
            # 取 topk 最大值索引
            maxk = min(self.topk, pred.size(1))
            _, indices = pred.topk(maxk, dim=1, largest=True, sorted=True)  # (B, topk)
            # 转置后第一行是 top1
            correct_matrix = indices.t().eq(label.view(1, -1).expand_as(indices.t()))
            # 只统计 top1
            self.correct += correct_matrix[:1].reshape(-1).float().sum().item()
            self.total += label.size(0)

    def compute(self) -> float:
        """返回百分制准确率"""
        return 100.0 * self.correct / self.total if self.total > 0 else 0.0