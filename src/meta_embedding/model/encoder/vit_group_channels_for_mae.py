# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

from .vit_group_channels import GroupChannelsVisionTransformer
from .pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from einops import rearrange, repeat

class GroupChannelViTForMae(GroupChannelsVisionTransformer):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, mask_ratio: float, spatial_mask: bool,
                 decoder_embed_dim: int, decoder_depth: int,
                 decoder_channel_embed: int=128, decoder_num_heads:int=16, decoder_mlp_ratio:float=4,
                 **kwargs):
        super().__init__(**kwargs)
        # Mask Strategy
        self.mask_ratio = mask_ratio
        self.spatial_mask = spatial_mask

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        num_patches = self.patch_embed[0].num_patches
        num_groups = len(self.channel_groups)
        # add pos embed in decoder
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim - decoder_channel_embed))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed[0].num_patches ** .5), cls_token=True)
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

        self.decoder_pred = nn.ModuleList([nn.Linear(decoder_embed_dim, len(group) * self.patch_size ** 2)
                                           for group in self.channel_groups])
        # --------------------------------------------------------------------------

        self.init_weights()

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):

        x = self.forward_before_mask(x)
        b, G, L, D = x.shape
        # ---------------------add mask
        if self.spatial_mask:
            # Mask spatial location across all channels (i.e. spatial location as either all/no channels)
            x = x.permute(0, 2, 1, 3).reshape(b, L, -1)  # (N, L, G*D)
            x, mask, ids_restore = self.random_masking(x, mask_ratio)  # (N, 0.25*L, G*D)
            x = x.view(b, x.shape[1], G, D).permute(0, 2, 1, 3).reshape(b, -1, D)  # (N, 0.25*G*L, D)
            mask = mask.repeat(1, G)  # (N, G*L)
            mask = mask.view(b, G, L)
        else:
            # Independently mask each channel (i.e. spatial location has subset of channels visible)
            x, mask, ids_restore = self.random_masking(x.view(b, -1, D), mask_ratio)  # (N, 0.25*G*L, D)
            mask = mask.view(b, G, L)
        h = w = int(L**0.5)
        mask = repeat(mask, "b g (h w) -> b g (h p1) (w p2)", h=h, w=w, p1=self.patch_size, p2=self.patch_size)
        in_chans = sum(len(group) for group in self.channel_groups)
        result_mask = torch.zeros(b, in_chans, self.img_size, self.img_size).to(mask.device)
        for i, group in enumerate(self.channel_groups):
            for band in group:
                result_mask[:,band] = mask[:, i]
        # ---------------------add mask

        x = self.forward_after_mask(x)

        return x, result_mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)  # (N, 1 + G*0.25*L, D)

        # append mask tokens to sequence
        G = len(self.channel_groups)
        if self.spatial_mask:
            N, L = ids_restore.shape

            x_ = x[:, 1:, :].view(N, G, -1, x.shape[2]).permute(0, 2, 1, 3)  # (N, 0.25*L, G, D)
            _, ml, _, D = x_.shape
            x_ = x_.reshape(N, ml, G * D)  # (N, 0.25*L, G*D)

            mask_tokens = self.mask_token.repeat(N, L - ml, G)
            x_ = torch.cat((x_, mask_tokens), dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))  # (N, L, G*D)
            x_ = x_.view(N, L, G, D).permute(0, 2, 1, 3).reshape(N, -1, D)  # (N, G*L, D)
            x = torch.cat((x[:, :1, :], x_), dim=1)  # append cls token  (N, 1 + G*L, D)
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token  (N, 1 + c*L, D)


        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # Separate channel axis
        N, GL, D = x.shape
        x = x.view(N, G, GL//G, D)

        # predictor projection
        x_c_patch = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, i]  # (N, L, D)
            dec = self.decoder_pred[i](x_c)  # (N, L, g_c * p^2)
            dec = dec.view(N, x_c.shape[1], -1, int(self.patch_size**2))  # (N, L, g_c, p^2)
            dec = torch.einsum('nlcp->nclp', dec)  # (N, g_c, L, p^2)
            x_c_patch.append(dec)

        x = torch.cat(x_c_patch, dim=1)  # (N, c, L, p**2)
        b, c, l, pp = x.shape
        h=w=int(l**0.5)
        p=int(pp**0.5)
        x = rearrange(x, "b c (h w) (p1 p2) -> b c (h p1) (w p2)", h=h,w=w,p1=p,p2=p)
        return x

    def forward(self, batch, is_pretrain: bool, is_classify: bool=None):
        if is_pretrain:
            latent, mask, ids_restore = self.forward_encoder(batch["x"], self.mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, C, L, p*p]
            return pred, mask


class GroupViTForMaeBaseDec512D1(GroupChannelViTForMae):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6),
                         decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
                         **kwargs)


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = GroupChannelViTForMae(
        channel_embed=256, embed_dim=1024, depth=24, num_heads=16,
        decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
