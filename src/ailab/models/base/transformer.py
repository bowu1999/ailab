import torch
import torch.nn as nn
import torch.nn.functional as F


class LabAttention(nn.Module):
    """Multi-head Self-Attention with optional fused kernel, q/k norm, proj/dropout."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        fused_attn: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # 是否使用 PyTorch scaled_dot_product_attention fused kernel
        self.fused_attn = fused_attn
        # 三合一线性映射 QKV
        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        # 可选的 per-head q/k 归一化
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias = proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B, N, C = x.shape
        # 产生 q, k, v 三个张量并 reshape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)
        # 可选归一化
        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            # PyTorch 2.0+ 原生 fused attention
            # [B, heads, N, head_dim]
            x = F.scaled_dot_product_attention(q, k, v, dropout_p = self.attn_drop.p if self.training else 0.)
        else:
            # 传统实现
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # [B, heads, N, head_dim]

        # 合并 heads → [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)
        # 输出投影 + dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LabMlp(nn.Module):
    """MLP block: FC → Activation → Dropout → (Optional Norm) → FC → Dropout."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: type = nn.GELU,
        drop: float = 0.,
        norm_layer: type = None,
        bias: bool = True,  # 添加这一行
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)  # 使用 bias 参数
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)  # 使用 bias 参数
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def lab_drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    DropPath (Stochastic Depth) per sample. Works on any tensor shape.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # shape = [B,1,1,...] 用于广播
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 0/1 mask
    if keep_prob > 0. and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class LabDropPath(nn.Module):
    """DropPath wrapper.
    DropPath 包装器已正确实现为 nn.Module 子类，该子类继承了 PyTorch Module 基类的训练标志。
    它在内部调用 drop_path 函数例程——该例程在训练期间随机丢弃特定于样本的残差分支（随机深度），
    并重新调整剩余路径以保留预期激活值。这种正则化已被证明可以通过防止路径相互适应来提高超深网络的泛化能力，
    并且最初由 Huang 等人（2016）以“随机深度”的形式引入。
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return lab_drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class LabLayerScale(nn.Module):
    """
    对残差分支输出作可学习缩放：y = gamma * x。
    init_values 通常为 1e-5 或 1e-6。
    """
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        # 每个通道一个缩放参数
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma



class LabAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        mlp_layer: type[nn.Module] = LabMlp,
    ) -> None:
        super().__init__()
        # 1st sublayer: LayerNorm → Attention → optional LayerScale → DropPath → residual
        self.norm1 = norm_layer(dim)
        self.attn = LabAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LabLayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = LabDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 2nd sublayer: LayerNorm → MLP → optional LayerScale → DropPath → residual
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LabLayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = LabDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block with residual
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # MLP block with residual
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x
