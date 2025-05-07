from abc import ABC, abstractmethod

class LabBaseMetric(ABC):
    """
    抽象 Metric 基类：
      - update(**kwargs): 接收本 batch 的字段（如 pred, label）并累积内部状态
      - compute(): 返回最终指标
      - reset(): 清空状态
    """
    def __init__(self, **cfg):
        self.cfg = cfg
        self.reset()

    @abstractmethod
    def update(self, **kwargs):
        ...

    @abstractmethod
    def compute(self):
        ...

    def reset(self):
        # 默认不做任何操作，子类可 override 初始化状态
        pass

    @property
    def avg(self):
        return self.compute()