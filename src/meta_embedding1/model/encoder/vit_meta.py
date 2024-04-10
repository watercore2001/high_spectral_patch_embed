from functools import partial

import torch
from einops import rearrange
from timm.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer
from torch import nn

__all__ = ["MetaViTBase", "MetaViTLarge"]


class MetaVisionTransformer(VisionTransformer):
    """ Vision Transformer with meta patch embedding
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.patch_embed
        img_size = kwargs["img_size"]
        self.in_chans = kwargs["in_chans"]
        embed_dim = kwargs["embed_dim"]
        self.patch_size = kwargs["patch_size"]
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=1, embed_dim=embed_dim)
            for _ in range(self.in_chans)
        ])
        self.fusion_channels = nn.Linear(in_features=self.in_chans * embed_dim,
                                         out_features=embed_dim)
        self.init_weights()

    def forward_features(self, x):
        # ---------- meta patch embed -----------
        x = torch.split(x, split_size_or_sections=1, dim=1)
        x = [embedding_layer(channel_x) for embedding_layer, channel_x in zip(self.patch_embeds, x)]
        # fusion all channels
        x = rearrange(x, pattern="c b l d -> b l (c d)")
        x = self.fusion_channels(x)
        # ---------- meta patch embed -----------

        # same as timm
        x = self._pos_embed(x)  # add cls token in this function
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)

        return x

    def forward(self, batch):
        return [super().forward(batch["x"])]


class MetaViTBase(MetaVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


class MetaViTLarge(MetaVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
