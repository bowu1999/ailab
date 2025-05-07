from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class LabBaseLoss(torch.nn.Module, ABC):
    """
    抽象 Loss 基类：
      - 子类需实现 forward(**kwargs)，声明需要哪些字段（如 pred, target, mask...）
      - 框架通过 call_fn 自动绑定 data_dict 中对应键
    """
    def __init__(self, **cfg):
        super().__init__()
        # 可在此根据 cfg 初始化超参数，比如权重、reduction 等
        self.cfg = cfg

    @abstractmethod
    def forward(self, **kwargs):
        """
        子类实现：明确需要哪些参数，
        例如 forward(self, pred, target, mask=None).
        """
        ...


class DynamicWeightedLoss(LabBaseLoss):
    """
    多头动态加权 Loss：
      - 子类实现 calculate_loss(**kwargs), 返回 {name: Tensor}
      - forward(**kwargs) 会自动调用 calculate_loss，并根据策略动态加权
    """
    def __init__(self,
                 weight_strategy: str = 'auto',
                 **cfg):
        super().__init__(weight_strategy=weight_strategy, **cfg)
        self.weight_strategy = weight_strategy

        # 不同策略的内部状态
        if weight_strategy == 'learnable':
            self.loss_weights = None       # 后续初始化为 nn.ParameterDict
        elif weight_strategy == 'fixed':
            self.fixed_weights = cfg.get('weights', None)
            assert self.fixed_weights is not None, "fixed 模式需传入 weights=dict(...)"
        elif weight_strategy == 'moving_avg':
            self.alpha = cfg.get('alpha', 0.5)
            self.avg_losses = {}
        elif weight_strategy == 'uncertainty':
            self.log_vars = nn.ParameterDict()
        # 'auto' 与其他策略无需额外初始化

    @abstractmethod
    def calculate_loss(self, **kwargs) -> dict:
        """
        子类实现：从 kwargs 中取必要字段（如 pred, target, mask…），
        返回 {loss_name: loss_tensor}
        """
        ...

    def get_weights(self, losses: dict) -> dict:
        """根据 self.weight_strategy 和当前 losses 值，返回同名 weight dict"""
        names = list(losses.keys())

        if self.weight_strategy == 'equal':
            return {n: 1.0 for n in names}

        if self.weight_strategy == 'fixed':
            return self.fixed_weights

        if self.weight_strategy == 'learnable':
            if self.loss_weights is None:
                # 第一次调用时创建可学习参数
                self.loss_weights = nn.ParameterDict({
                    n: nn.Parameter(torch.tensor(0.0)) for n in names
                })
            return {n: torch.exp(self.loss_weights[n]) for n in names}

        # auto / moving_avg
        loss_vals = {n: float(v.item()) for n, v in losses.items()}
        if self.weight_strategy == 'moving_avg':
            for n in names:
                self.avg_losses[n] = (self.avg_losses.get(n, loss_vals[n]) * self.alpha
                                      + loss_vals[n] * (1 - self.alpha))
            loss_vals = self.avg_losses

        invs = {n: 1.0 / (loss_vals[n] + 1e-8) for n in names}
        norm = sum(invs.values())
        return {n: invs[n] / norm for n in names}

    def forward(self, **kwargs) -> torch.Tensor:
        """
        框架会根据 mapping，把 data_dict 中的各键传给对应参数名
        例如 forward(pred=…, target=…, mask=…)
        """
        # 1) 计算各子 loss
        losses = self.calculate_loss(**kwargs)
        # 2) 根据策略决定权重
        weights = self.get_weights(losses)
        # 3) 加权求和
        total = sum(losses[n] * weights[n] for n in losses)
        return total
