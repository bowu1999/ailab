import torch
import torch.nn as nn
from einops import rearrange

from .transformer import LabAttentionBlock


# -- 1.1 ViT Patch Embedding --
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B,3,H,W] → [B,embed_dim,num_patches_sqrt,num_patches_sqrt]
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


# -- 1.2 Encoder (asymmetric, only visible tokens) --
class MAEEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        # learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            LabAttentionBlock(embed_dim, num_heads, mlp_ratio=4.) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # x: [B,3,H,W]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        B, N, C = x.shape
        # If no mask provided (inference), treat all tokens as visible
        if mask is None:
            # Create a false mask: no tokens are masked
            mask = torch.zeros((B, N), device=x.device, dtype=torch.bool)
        else:
            # ensure tensor and bool type
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, device=x.device, dtype=torch.bool)
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            # support mask broadcast or reshape if needed
            mask = mask.view(B, N)
        # select only visible tokens
        # x[~mask] flattens batch, so reshape back
        x_visible = x[~mask].reshape(B, -1, C)

        # forward through transformer blocks
        for blk in self.blocks:
            x_visible = blk(x_visible)
        
        # final normalization
        return self.norm(x_visible)


# -- 1.3 Decoder (lightweight) --
class MAEDecoder(nn.Module):
    def __init__(self, num_patches, embed_dim=768, decoder_dim=512, depth=4, num_heads=8):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.proj = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.blocks = nn.ModuleList([
            LabAttentionBlock(decoder_dim, num_heads, mlp_ratio=4.) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, embed_dim, bias=True)

    def forward(self, x_enc, mask):
        B, visible, _ = x_enc.shape
        x = self.proj(x_enc)
        # expand mask tokens and interleave
        # mask_tokens = self.mask_token.expand(B, mask.sum(dim=1)[0], -1)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        num_masked = mask.sum(dim=1)[0].item()  # 第一个样本的被 mask 数（假设都一样）
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        x_full = torch.zeros(B, mask.shape[1], x.shape[2], device=x.device)
        # 将 mask 转换为布尔类型
        mask_bool = mask.to(torch.bool)
        # 使用 masked_scatter_ 将可见的 x 值填充到 x_full 中
        x_full.masked_scatter_(~mask_bool.unsqueeze(-1), x)
        # 使用 masked_scatter_ 将 mask_tokens 填充到 x_full 中
        x_full.masked_scatter_(mask_bool.unsqueeze(-1), mask_tokens)
        x_full = x_full + self.pos_embed_dec
        for blk in self.blocks:
            x_full = blk(x_full)
        return self.head(self.norm(x_full))


# -- 1.4 Full MAE with pretrain loading --
class MaskedAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = MAEEncoder(**kwargs)
        self.decoder = MAEDecoder(self.encoder.patch_embed.num_patches, **kwargs)

    def forward(self, image, mask):
        enc = self.encoder(image, mask)
        return self.decoder(enc, mask)

    def load_pretrained(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(state, strict=False)
        return self
