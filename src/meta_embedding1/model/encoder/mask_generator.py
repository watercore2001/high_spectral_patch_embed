from einops import repeat, rearrange
import torch
import numpy as np


def mask_in_group(groups: int, image_size: int, mask_unit:int, mask_ratio:float):
    mask_unit_num_one_side = image_size // mask_unit
    patch_num = groups * mask_unit_num_one_side ** 2
    mask_num = int(np.ceil(patch_num * mask_ratio))

    mask_idx = np.random.permutation(patch_num)[:mask_num]
    mask = np.zeros(patch_num, dtype=int)
    mask[mask_idx] = 1

    mask = repeat(mask, pattern="(c h w) -> c h w",
                  c=groups,
                  h=mask_unit_num_one_side,
                  w=mask_unit_num_one_side,)

    return torch.Tensor(mask)


class MaskGenerator:

    def mask_in_space(self):
        patch_num_pre_band = self.mask_patch_num_one_side ** 2
        mask_num = int(np.ceil(patch_num_pre_band * self.mask_ratio))

        mask_idx = np.random.permutation(patch_num_pre_band)[:mask_num]

        mask = np.zeros(patch_num_pre_band, dtype=int)
        mask[mask_idx] = 1

        mask = repeat(mask, pattern="(h w) -> c (h s1) (w s2)",
                      c=self.channels,
                      h=self.mask_patch_num_one_side,
                      w=self.mask_patch_num_one_side,
                      s1=self.scale,
                      s2=self.scale)
        return torch.Tensor(mask)

    def mask_in_band(self):
        mask_band_num = int(np.ceil(self.channels * self.mask_ratio))

        mask_idx = np.random.permutation(self.channels)[:mask_band_num]

        mask = np.zeros(self.channels, dtype=int)
        mask[mask_idx] = 1

        mask = repeat(mask, pattern="c -> c h w", h=self.model_patch_num_one_side, w=self.model_patch_num_one_side)
        return torch.Tensor(mask)

