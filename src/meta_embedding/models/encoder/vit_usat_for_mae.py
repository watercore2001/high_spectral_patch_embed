# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.vision_transformer import Block

from meta_embedding.models.encoder.util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from .util.mae import random_masking
from .vit_usat import USatVisionTransformer

__all__ = ["USatForMaeBaseDec512D1"]


class USatViTForMae(USatVisionTransformer):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, decoder_embed_dim: int, decoder_depth: int,
                 decoder_channel_embed: int = 128,
                 decoder_num_heads: int = 16,
                 decoder_mlp_ratio: float = 4,
                 mask_ratio: float = None,
                 **kwargs):
        super().__init__(**kwargs)
        # Mask Strategy
        self.mask_ratio = mask_ratio

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        num_patches = self.patch_embeds[0].num_patches
        num_groups = len(self.channel_groups)
        # add pos embed in decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim - decoder_channel_embed),
            requires_grad=False)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.decoder_channel_embed = nn.Parameter(torch.zeros(1, num_groups + 1, decoder_channel_embed),
                                                  requires_grad=False)
        dec_channel_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_channel_embed.shape[-1],
                                                              torch.arange(len(self.channel_groups) + 1).numpy())
        self.decoder_channel_embed.data.copy_(torch.from_numpy(dec_channel_embed).float().unsqueeze(0))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(decoder_depth)])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.decoder_preds = nn.ModuleList([nn.Linear(decoder_embed_dim, len(group) * (gsd//self.min_gsd*self.patch_size) ** 2)
                                           for group, gsd in zip(self.channel_groups, self.group_gsds)])
        # --------------------------------------------------------------------------
        self.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"decoder_pos_embed", "decoder_channel_embed"}.union(super().no_weight_decay())

    def forward_encoder(self, x, mask_ratio):
        b, c, h, w = x.shape
        result_mask = torch.zeros(b, c, h, w).to(x.device)

        x = self.forward_before_mask(x)  # [(b,h1,w1,d),(b,h2,w2,d),(b,h3,w3,d)]

        # Independently mask each channel (i.e. spatial location has subset of channels visible)
        x_list, x_lengths, ids_restore_list = [], [], []

        for i, x_i in enumerate(x):
            x_i = rearrange(x_i, 'b h w d -> b (h w) d')

            x_i, mask, ids_restore = random_masking(x_i, mask_ratio)
            _, l_i, _ = x_i.shape
            x_list.append(x_i)
            x_lengths.append(l_i)
            ids_restore_list.append(ids_restore)

            patch_size = self.group_gsds[i] // self.min_gsd * self.patch_size
            mask = repeat(mask, "b (h w) -> b (h p1) (w p2)", h=h//patch_size, w=w//patch_size,
                          p1=patch_size, p2=patch_size)
            for band in self.channel_groups[i]:
                result_mask[:, band] = mask

        x = torch.cat(x_list, dim=1)

        # ---------------------add mask
        x = self.forward_after_mask(x)

        return x, x_lengths, result_mask, ids_restore_list

    def forward_decoder(self, x, x_lengths, ids_restores):
        # embed tokens
        x = self.decoder_embed(x)  # (N, 1 + L, D)
        b, _, d = x.shape

        # append mask tokens to sequence
        start_idx = 1
        x_list = [x[:, 0:start_idx]]
        for i, length in enumerate(x_lengths):
            x_i = x[:, start_idx:start_idx + length]
            start_idx += length
            mask_tokens = self.mask_token.repeat(b, ids_restores[i].shape[1] - length, 1)
            x_i = torch.cat([x_i, mask_tokens], dim=1)
            x_i = torch.gather(x_i, dim=1, index=ids_restores[i].unsqueeze(-1).repeat(1, 1, d))
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)  # append cls token  (N, 1 + c*L, D)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        x = self.recovery(x)

        return x

    def recovery(self, x):
        _, L, _ = x.shape
        start_idx = 0
        out = []
        for i, gsd in enumerate(self.group_gsds):
            # print("start_idx:", start_idx)
            scale = gsd // self.min_gsd
            num_patches = self.num_patches // (scale ** 2)
            h = w = int(num_patches ** 0.5)
            patch_size = gsd//self.min_gsd*self.patch_size

            x_i = x[:, start_idx:start_idx+num_patches, :]
            start_idx += num_patches
            x_i = self.decoder_preds[i](x_i)
            x_i = rearrange(x_i, "b (h w) (g p1 p2) -> b g (h p1) (w p2)", g=len(self.channel_groups[i]),
                            h=h, w=w, p1=patch_size, p2=patch_size)
            out.append(x_i)

        assert start_idx == L, f"{start_idx}, {L}"
        out = torch.cat(out, dim=1)

        return out

    def forward(self, batch):
        latent, x_lengths, mask, ids_restore = self.forward_encoder(batch["x"], self.mask_ratio)
        pred = self.forward_decoder(latent, x_lengths, ids_restore)  # [N, C, L, p*p]
        return pred, mask


class USatForMaeBaseDec512D1(USatViTForMae):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6),
                         decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
                         **kwargs)
