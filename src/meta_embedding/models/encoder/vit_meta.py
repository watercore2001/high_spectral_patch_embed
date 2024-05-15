from functools import partial

from typing import Literal

import torch
from einops import rearrange
from timm.layers import PatchEmbed, trunc_normal_

import timm
from torch import nn

__all__ = ["MetaViTBase"]


class MetaVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with meta patch embedding
    """

    def __init__(self,
                 use_channel_embed: bool,
                 use_transformer: bool,
                 fusion_strategy: Literal["linear", "add", "avg", "max"],
                 channel_heads: int = None,
                 channel_dim: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        img_size = kwargs["img_size"]
        self.in_chans = kwargs["in_chans"]
        self.patch_size = kwargs["patch_size"]

        if channel_dim is None:
            channel_dim = self.embed_dim

        self.patch_embeds = nn.ModuleList([
            PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=1, embed_dim=channel_dim,
                       bias=use_channel_embed)
            for _ in range(self.in_chans)
        ])

        self.use_transformer = use_transformer
        if self.use_transformer:
            if channel_heads is None:
                channel_heads = self.embed_dim // 64
            self.spectral_transformer = nn.TransformerEncoderLayer(d_model=channel_dim, nhead=channel_heads,
                                                                   dim_feedforward=channel_dim * 4,
                                                                   batch_first=True)

        self.fusion_strategy = fusion_strategy
        if fusion_strategy == "linear":
            self.fusion_channels = nn.Linear(in_features=self.in_chans * channel_dim,
                                             out_features=self.embed_dim)
        if fusion_strategy == "add":
            self.fusion_channels = nn.Linear(in_features=self.in_chans,
                                             out_features=1)

        self.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"channel_embed"}.union(super().no_weight_decay())

    def forward_before_mask(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        x = [embedding_layer(channel_x) for embedding_layer, channel_x in zip(self.patch_embeds, x)]
        x = rearrange(x, pattern="c b l d -> b l c d")

        # use one transformer layer
        if self.use_transformer:
            b, l, _, _ = x.shape
            x = rearrange(x, pattern="b l c d -> (b l) c d")
            x = self.spectral_transformer(x)
            x = rearrange(x, pattern="(b l) c d -> b l c d", b=b, l=l)

        # fusion all channels
        # before fusion x: b l c d
        # after fusion x: b l d
        if self.fusion_strategy == "linear":
            x = rearrange(x, pattern="b l c d -> b l (c d)")
            x = self.fusion_channels(x)
        if self.fusion_strategy == "add":
            x = rearrange(x, pattern="b l c d -> b l d c")
            x = self.fusion_channels(x)
            x = rearrange(x, pattern="b l d 1 -> b l d")
        elif self.fusion_strategy == "avg":
            x = torch.mean(input=x, dim=2, keepdim=False)
        elif self.fusion_strategy == "max":
            x = torch.max(input=x, dim=2, keepdim=False)

        return x

    def forward_after_mask(self, x):
        # same as timm
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)

        return x

    def forward_features(self, x):
        x = self.forward_before_mask(x)
        x = self._pos_embed(x)  # add cls token in this function
        x = self.forward_after_mask(x)

        return x

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


class MetaViTBase(MetaVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
