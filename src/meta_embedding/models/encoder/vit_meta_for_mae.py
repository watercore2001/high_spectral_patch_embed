from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

from .util.mae import random_masking
from .vit_meta import MetaVisionTransformer
from typing import Literal

__all__ = ["MetaViTForMaeBaseDec512D1"]


class MetaViTForMae(MetaVisionTransformer):
    def __init__(self,
                 decoder_embed_dim: int, decoder_depth: int, decoder_num_heads: int = 16,
                 decoder_mlp_ratio: float = 4, mask_ratio: float = None, **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio

        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        num_patches = self.patch_embed[0].num_patches
        # add pos embed in decoder
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, decoder_embed_dim) * .02)
        trunc_normal_(self.decoder_pos_embed, std=.02)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(decoder_depth)])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size ** 2 * self.in_chans, bias=True)

        self.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"decoder_pos_embed"}.union(super().no_weight_decay())

    def forward_encoder(self, x, mask_ratio):
        x = self.forward_before_mask(x)

        B, L, C = x.shape

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = random_masking(x, mask_ratio)

        h = w = int(L ** 0.5)
        result_mask = repeat(mask, "b (h w) -> b c (h p1) (w p2)", c=self.in_chans, h=h, w=w, p1=self.patch_size,
                             p2=self.patch_size)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.forward_after_mask(x)

        return x, result_mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        b, l, c_pp = x.shape
        h = w = int(l ** 0.5)
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", h=h, w=w, p1=self.patch_size, p2=self.patch_size)
        return x

    def forward(self, batch):
        latent, mask, ids_restore = self.forward_encoder(batch["x"], self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [b, c, h, w]
        return pred, mask


class MetaViTForMaeBaseDec512D1(MetaViTForMae):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_embed_dim=512, decoder_depth=1,
                         decoder_num_heads=16, **kwargs)
