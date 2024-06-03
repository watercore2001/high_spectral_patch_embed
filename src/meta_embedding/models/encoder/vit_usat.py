# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from einops import rearrange, reduce, repeat

from meta_embedding.models.encoder.util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid

__all__ = ["USatVisionTransformer", "USatViTBase"]


def get_gsd(id, channel_groups, group_gsds):
    for i, group in enumerate(channel_groups):
        if id in group:
            return group_gsds[i]

class USatVisionTransformer(timm.models.vision_transformer.VisionTransformer):

    def __init__(self, channel_embed=256, channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),
                 group_gsds=(10, 20, 20), **kwargs):
        super().__init__(**kwargs)
        self.in_chans = kwargs["in_chans"]
        self.img_size = kwargs['img_size']
        self.patch_size = kwargs['patch_size']
        self.embed_dim = kwargs['embed_dim']

        self.channel_groups = channel_groups
        self.group_gsds = group_gsds


        patch_embed_list = []
        self.min_gsd = min(group_gsds)
        for i in range(self.in_chans):
            gsd = get_gsd(i, channel_groups, group_gsds)
            patch_size = gsd//self.min_gsd*self.patch_size
            patch_embed_list.append(PatchEmbed(self.img_size, patch_size, 1, self.embed_dim, output_fmt="NHWC"))

        self.patch_embeds = nn.ModuleList(patch_embed_list)

        num_patches = self.patch_embeds[0].num_patches
        self.num_patches = num_patches

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

        self.fc = nn.Linear(num_groups, 1)

        self.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"channel_embed", "channel_cls_embed"}.union(super().no_weight_decay())

    def forward_before_mask(self, x):
        # ---------- patch embed -----------
        x = torch.split(x, split_size_or_sections=1, dim=1)
        x = [embedding_layer(channel_x) for embedding_layer, channel_x in zip(self.patch_embeds, x)]
        # 将每一组的波段取出并计算平均值
        grouped_tensors = []
        for group in self.channel_groups:
            # 从第1维度选择对应波段
            selected_tensor = [x[i] for i in group]
            # 对第1维度求平均值
            mean_tensor = torch.stack(selected_tensor).mean(dim=0, keepdim=False)
            grouped_tensors.append(mean_tensor)

        # [(b,h1,w1,d),(b,h2,w2,d),(b,h3,w3,d)]
        # ----------------------------------

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, G, 1, cD)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        # Channel embed same across (x,y) position, and pos embed same across channel (g)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, G, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, G, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)
        _, _, l, _ = pos_channel.shape
        h = w = int(l**0.5)
        pos_channel = rearrange(pos_channel, "1 g (h w) d -> g h w d", h=h, w=w)
        pos_channel = torch.split(pos_channel, split_size_or_sections=1, dim=0)
        # add pos embed w/o cls token
        result = []
        for x_i, pos_channel_i in zip(grouped_tensors, pos_channel):
            _, h_i, w_i, _ = x_i.shape
            scale = h//h_i
            pos_channel_i = reduce(pos_channel_i, "1 (h scale1) (w scale2) d -> 1 h w d", "mean",
                                   scale1=scale, scale2=scale)
            result.append(x_i+pos_channel_i)

        return result

    def forward_after_mask(self, x):
        b, _, _ = x.shape
        cls_pos_channel = torch.cat((self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)  # (1, 1, D)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, 1 + G*L, D)
        x = self.pos_drop(x)

        # same as timm
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_features(self, x):
        x = self.forward_before_mask(x)
        merged_tensors = []
        for tensor in x:
            # 将 h 和 w 合并
            merged_tensor = rearrange(tensor, "b h w d -> b (h w) d")
            merged_tensors.append(merged_tensor)

        # 按列拼接所有合并后的 Tensor
        x = torch.cat(merged_tensors, dim=1)

        x = self.forward_after_mask(x)

        return x

    def resample(self, x):
        _, L, _ = x.shape
        start_idx = 0
        out = []
        for gsd in self.group_gsds:
            # print("start_idx:", start_idx)
            scale = gsd // self.min_gsd
            num_patches = self.num_patches // (scale ** 2)
            h = w = int(num_patches ** 0.5)
            x_i = x[:, start_idx:start_idx+num_patches, :]
            start_idx += num_patches
            x_i = rearrange(x_i, "b (h w) d -> b h w d", h=h, w=w)
            x_i = repeat(x_i, "b h w d -> b (h scale1) (w scale2) d", scale1=scale, scale2=scale)
            out.append(x_i)

        assert start_idx == L, f"{start_idx}, {L}"
        out = torch.stack(out)
        out = rearrange(out, "g b h w d -> b d h w g")
        out = self.fc(out)
        out = rearrange(out, "b d h w 1 -> b d h w")
        return out


    def forward(self, batch, is_classify: bool):
        if is_classify:
            return super().forward(batch["x"])
        else:
            x = self.forward_features(batch["x"])
            x = x[:, 1:, :] # [b L d]
            return self.resample(x)


class USatViTBase(USatVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
