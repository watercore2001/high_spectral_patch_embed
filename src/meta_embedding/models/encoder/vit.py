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

__all__ = ["VisionTransformer", "ViTBase"]


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, batch):
        return [super().forward(batch["x"])]


class ViTBase(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
