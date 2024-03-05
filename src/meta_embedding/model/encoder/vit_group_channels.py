# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed
from .pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid

__all__ = ["GroupViTBase", "GroupViTLarge"]


class GroupChannelsVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, channel_embed=256, channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)), **kwargs):
        super().__init__(**kwargs)
        self.img_size = kwargs['img_size']
        self.patch_size = kwargs['patch_size']
        self.embed_dim = kwargs['embed_dim']

        self.channel_groups = channel_groups

        self.patch_embed = nn.ModuleList([PatchEmbed(self.img_size, self.patch_size, len(group), self.embed_dim)
                                          for group in channel_groups])
        # self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed[0].num_patches

        # Positional and channel embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim - channel_embed))
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        num_groups = len(channel_groups)
        self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed))
        chan_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], torch.arange(num_groups).numpy())
        self.channel_embed.data.copy_(torch.from_numpy(chan_embed).float().unsqueeze(0))

        # Extra embedding for cls to fill embed_dim
        self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))
        channel_cls_embed = torch.zeros((1, channel_embed))
        self.channel_cls_embed.data.copy_(channel_cls_embed.float().unsqueeze(0))

        self.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self) :
        return {'channel_embed', 'channel_cls_embed'}.union(super().no_weight_decay())

    def forward_before_mask(self, x):
        # ---------- start 1. patch embed -----------
        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(self.patch_embed[i](x_c))  # (N, L, D)

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
        # ---------- end 1. patch embed -----------

        _, G, L, D = x.shape

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, G, 1, cD)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        # Channel embed same across (x,y) position, and pos embed same across channel (g)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, G, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, G, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)

        # add pos embed w/o cls token
        x = x + pos_channel  # (N, G, L, D)
        return x

    def forward_after_mask(self, x):
        b, _, _ = x.shape
        cls_pos_channel = torch.cat((self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)  # (1, 1, D)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, 1 + G*L, D)

        x = self.pos_drop(x)
        # ---------- end 2. pos embedding -----------

        # same as timm
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_features(self, x):
        x = self.forward_before_mask(x)
        b, G, L, D = x.shape
        x = x.view(b, -1, D)  # (N, G*L, D)
        x = self.forward_after_mask(x)

        return x

    def forward(self, batch):
        return [super().forward(batch["x"])]


class GroupViTBase(GroupChannelsVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


class GroupViTLarge(GroupChannelsVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)