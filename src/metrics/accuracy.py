import torch
from ._base import Metric


class Accuracy(Metric):
    def __init__(self, topk=1):
        self.topk = topk
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        with torch.no_grad():
            maxk = min(self.topk, outputs.size(1))
            _, pred = outputs.topk(maxk, 1, True, True)  # (N, topk)
            pred = pred.t()                               # (topk, N)
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            # 只统计 top1 准确率
            self.correct += correct[:1].reshape(-1).float().sum().item()
            self.total += targets.size(0)

    def compute(self):
        return 100.0 * self.correct / self.total if self.total > 0 else 0.0
    
    @property
    def avg(self):
        """方便 LoggerHook 直接读 .avg 获取当前平均值"""
        return self.compute()