# --------------------------------------------------------
# References:
# SatMAE: https://github.com/sustainlab-group/SatMAE
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
from torch import nn
from einops import rearrange

__all__ = ["VisionTransformer", "ViTBase"]


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, batch, is_classify: bool):
        if is_classify:
            return super().forward(batch["x"])
        else:
            x = self.forward_features(batch["x"]) # [b l d]
            x = x[:, 1:, :]
            _, l, _ = x.shape
            h = w = int(l ** 0.5)
            x = rearrange(x, "b (h w) d -> b d h w", h=h, w=w)
            return x


class ViTBase(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
