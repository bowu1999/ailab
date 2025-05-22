import inspect
import torch.nn as nn


class OutputWrapper(nn.Module):
    """
    在 __init__ 中动态生成一个名为 forward 的方法，其签名 =
    wrap_model.forward 的签名，以便 call_fn 能正确 bind 参数。
    """

    def __init__(self, model: nn.Module, output_keys=None):
        super().__init__()
        self.model = model
        self.output_keys = output_keys or ["output"]

        # 取底层 model.forward 签名
        orig_sig = inspect.signature(model.forward)

        # 定义一个新的 forward_impl，绑定到实例
        def forward_impl(*args, **kwargs):
            # 调用原模型
            out = model(*args, **kwargs)
            # 规范化输出为 dict
            if isinstance(out, dict):
                return out
            if isinstance(out, (tuple)):
                return dict(zip(self.output_keys, out))
            return {self.output_keys[0]: out}

        # 让 inspect.signature(OutputWrapper.forward) 返回 orig_sig
        forward_impl.__signature__ = orig_sig

        # 将实例的 forward 绑定到 forward_impl
        # 这样 DDP.wrap 的就是这个有正确签名的 forward_impl
        object.__setattr__(self, "forward", forward_impl)



def call_fn(fn, data_dict, mapping=None):
    """
    通用调用函数：
     - mapping: dict, key=fn参数名, value=data_dict中的字段名
       e.g. mapping = {'preds':'pred', 'targets':'target'}
     - 会根据 fn 的签名，给位置参数和 keyword-only 参数正确取值，
       未在 mapping 里声明的，就直接按参数名去 data_dict 中找 key。
    """
    # 如果是 DDP 包装，真正签名在 .module.forward
    target = getattr(fn, 'module', fn)
    sig = inspect.signature(getattr(target, 'forward', target))

    args, kwargs = [], {}
    missing = []

    for name, param in sig.parameters.items():
        if name == 'self':
            continue

        # 先看 mapping (param → data_key)，否则 data_key = name
        data_key = mapping.get(name, name) if mapping else name

        if data_key in data_dict:
            val = data_dict[data_key]
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD):
                args.append(val)
            else:  # KEYWORD_ONLY 或 VAR_KEYWORD
                kwargs[name] = val
        else:
            # 如果是必需参数且没 default，就报错
            if (param.default is inspect._empty
                and param.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                       inspect.Parameter.VAR_KEYWORD)):
                missing.append(name)

    if missing:
        raise ValueError(f"Missing required fields {missing} for {fn}")

    return fn(*args, **kwargs)
