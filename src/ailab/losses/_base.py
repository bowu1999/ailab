from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class DynamicWeightedLoss(nn.Module, ABC):
    def __init__(self, weight_strategy = 'auto', **kwargs):
        """
        Args:
            weight_strategy: 权重策略：
                - 'auto': 基于损失大小自动调整（默认）
                - 'learnable': 可学习的权重参数
                - 'fixed': 手动指定固定权重（需要传入 weights=dict(...)）
                - 'equal': 所有权重设为1.0
        """
        super(DynamicWeightedLoss, self).__init__()
        self.weight_strategy = weight_strategy
        self.kwargs = kwargs
        
        # 如果是 learnable 策略，则由抽象类统一管理
        if weight_strategy == 'learnable':
            self.loss_weights = None  # 会在第一次 calculate_loss 后初始化
        elif weight_strategy == 'fixed':
            self.fixed_weights = kwargs.get('weights', None)
            assert self.fixed_weights is not None, "请提供固定权重字典: weights={...}"
        elif weight_strategy == 'moving_avg':
            self.alpha = kwargs.get('alpha', 0.5)
            self.avg_losses = dict()
        elif weight_strategy == 'uncertainty':
            self.log_vars = nn.ParameterDict()
        else:
            # auto 或其它未特别配置的情况
            pass

    @abstractmethod
    def calculate_loss(self, outputs, targets):
        """
        子类必须实现该方法，返回以 loss_name 为键的损失字典
        
        Args:
            outputs: 模型输出，dict or tuple
            targets: 真实标签，dict or tuple
            
        Returns:
            dict: {loss_name: loss_tensor}
        """
        pass

    def get_weights(self, losses):
        """
        根据当前损失值获取各任务的权重，封装了多种策略
        """
        names = list(losses.keys())
        
        if self.weight_strategy == 'equal':
            return {name: 1.0 for name in names}

        elif self.weight_strategy == 'fixed':
            return self.fixed_weights

        elif self.weight_strategy == 'learnable':
            # 第一次调用时，根据 loss 名称创建可学习参数
            if self.loss_weights is None:
                self.loss_weights = nn.ParameterDict({
                    name: nn.Parameter(torch.tensor(0.0)) for name in names
                })
            return {name: torch.exp(param) for name, param in self.loss_weights.items()}

        elif self.weight_strategy == 'auto' or self.weight_strategy == 'moving_avg':
            # 自动基于当前损失大小反比加权
            loss_values = {k: float(v.item()) for k, v in losses.items()}
            
            if self.weight_strategy == 'moving_avg':
                # 使用指数平滑移动平均
                for name in names:
                    if name not in self.avg_losses:
                        self.avg_losses[name] = loss_values[name]
                    else:
                        self.avg_losses[name] = \
                            self.alpha * self.avg_losses[name] + (1 - self.alpha) * loss_values[name]
                loss_values = self.avg_losses

            # 归一化权重：w_i = 1 / L_i
            total_inv = sum(1.0 / (loss_values[name] + 1e-8) for name in names)
            weights = {name: (1.0 / (loss_values[name] + 1e-8)) / total_inv for name in names}
            return weights

        elif self.weight_strategy == 'uncertainty':
            # 基于不确定性加权（每个任务一个 log_var 参数）
            for name in names:
                if name not in self.log_vars:
                    self.log_vars[name] = nn.Parameter(torch.tensor(0.0))
            precision_terms = {name: torch.exp(-self.log_vars[name]) for name in names}
            return precision_terms

        else:
            raise ValueError(f"不支持的权重策略: {self.weight_strategy}")

    def forward(self, outputs, targets):
        # 计算各个头的损失
        losses = self.calculate_loss(outputs, targets)
        # 获取动态权重
        weights = self.get_weights(losses)
        # 加权求和
        total_loss = sum(weight * losses[name] for name, weight in weights.items())

        return total_loss
