from functools import partial

import torch
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import init_weights_vit_timm
from torch import nn

from .vit_meta import MetaVisionTransformer

__all__ = ["MetaViTForSatMIMBase", "MetaViTForSatMIMLarge"]


class MetaVisionTransformerForSatMIM(MetaVisionTransformer):
    """ Vision Transformer with meta patch embedding
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_chans = kwargs["in_chans"]
        embed_dim = kwargs["embed_dim"]

        # initial mask tokens for each channel
        self.mask_tokens = nn.Parameter(torch.zeros(in_chans, embed_dim))
        trunc_normal_(self.mask_tokens, mean=0., std=.02)

        # recovery size
        self.recovery_header = nn.Linear(embed_dim, in_chans * self.patch_size ** 2)
        init_weights_vit_timm(self.recovery_header)

    def forward_features(self, x, mask):
        b, c, h, w = x.shape
        # ---------- 1. patch embedding for each channel -----------
        x = torch.split(x, split_size_or_sections=1, dim=1)
        x = [embedding_layer(channel_x) for embedding_layer, channel_x in zip(self.patch_embeds, x)]
        new_h, new_w = h // self.patch_size, w // self.patch_size
        x = rearrange(x, pattern="c b l d -> b c l d")
        # ---------- 2. add mask -----------
        if mask != None:
            mask = rearrange(mask, pattern="b c new_h new_w -> b c (new_h new_w) 1")
            mask_tokens = repeat(self.mask_tokens, pattern="c d -> b c (new_h new_w) d", b=b, new_h=new_h, new_w=new_w)
            x = x * (1 - mask) + mask_tokens * mask
        # ---------- 3. fusion all channels -----------
        x = rearrange(x, pattern="b c l d -> b l (c d)")
        x = self.fusion_channels(x)

        # ---------- 4. same as timm -----------
        x = self._pos_embed(x)  # add cls token in this function
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, batch, is_pretrain: bool, is_classify: bool):
        b, c, h, w = batch["x"].shape
        new_h, new_w = h // self.patch_size, w // self.patch_size

        if is_pretrain:
            x = self.forward_features(batch["x"], batch["mask"])
            x = x[:, 1:, :]
            x = self.recovery_header(x)
            x = rearrange(x, "b (new_h new_w) (c patch_size_h patch_size_w) -> "
                             "b c (new_h patch_size_h) (new_w patch_size_w)",
                          patch_size_h=self.patch_size, patch_size_w=self.patch_size,
                          new_h=new_h, new_w=new_w)
            return x
        elif is_classify:
            x = self.forward_features(batch["x"], batch["mask"])
            x = self.forward_head(x)
            return [x]
        else:
            x = self.forward_features(batch["x"], batch["mask"])
            x = rearrange(x, "b (new_h new_w) c -> b c new_h new_w", new_h=new_h, new_w=new_w)
            return [x]


class MetaViTForSatMIMBase(MetaVisionTransformerForSatMIM):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


class MetaViTForSatMIMLarge(MetaVisionTransformerForSatMIM):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
