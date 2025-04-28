class Metric:
    """指标基类：必须实现 reset, update, compute"""
    def reset(self):
        raise NotImplementedError
    def update(self, outputs, targets):
        raise NotImplementedError
    def compute(self):
        raise NotImplementedError