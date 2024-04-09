from .swint import SwinTransformer
from torch import nn
from timm.layers import PatchEmbed
import torch
from einops import rearrange

class MetaSwinTransformer(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.patch_embed
        img_size = kwargs["img_size"]
        self.in_chans = kwargs["in_chans"]
        embed_dim = kwargs["embed_dim"]
        self.patch_size = kwargs["patch_size"]
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=1, embed_dim=embed_dim, output_fmt='NHWC')
            for _ in range(self.in_chans)
        ])

        self.fusion_channels = nn.Linear(in_features=self.in_chans*embed_dim,
                                         out_features=embed_dim)
        self.init_weights()

    def forward_features(self, x):
        # ---------- meta patch embed -----------
        x = torch.split(x, split_size_or_sections=1, dim=1)
        x = [embedding_layer(channel_x) for embedding_layer, channel_x in zip(self.patch_embeds, x)]
        # fusion all channels
        x = rearrange(x, pattern="c b h w d -> b (h w) (c d)")
        x = self.fusion_channels(x)
        _, l, _ = x.shape
        h = w = int(l**0.5)
        x = rearrange(x, pattern="b (h w) c -> b h w c", h=h, w=w)
        # ---------- meta patch embed -----------

        #  same as timm
        x = self.layers(x)
        x = self.norm(x)
        return x

class MetaSwinBase(MetaSwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=4, window_size=8, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
                         **kwargs)



