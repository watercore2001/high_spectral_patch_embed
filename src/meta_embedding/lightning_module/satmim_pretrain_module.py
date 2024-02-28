import os.path
from typing import Any
from typing import Literal
from torch import nn
from einops import repeat, rearrange
import torch
from .base_module import AdamWCosineOptimArgs, BaseModule
import numpy as np


__all__ = ["SatMIMPreTrainingModule"]


class SatMIMPreTrainingModule(BaseModule):
    def __init__(self, optim_args: AdamWCosineOptimArgs, encoder: nn.Module,
                 image_size: int, channels: int, mask_patch_size: int, model_patch_size: int, mask_ratio: float):
        super().__init__(optim_args=optim_args, encoder=encoder, decoder=None, header=None)
        self.mask_generator = MaskGenerator(image_size, channels, mask_patch_size, model_patch_size, mask_ratio)
        self.patch_size = model_patch_size
        self.channels = channels
        self.image_size = image_size
        self.l1_loss = nn.L1Loss(reduction="none")

    def forward(self, batch: dict):
        x = batch["x"]
        b, c, h, w = x.shape
        assert h == self.image_size and w == self.image_size
        assert c == self.channels
        masks = []
        for i in range(b):
            masks.append(self.mask_generator(strategy="random"))
        mask = rearrange(masks, 'b c new_h new_w -> b c new_h new_w').to(x.device)
        batch["mask"] = mask
        x_recover = self.encoder(batch, is_pretrain=True)
        all_loss = self.l1_loss(x_recover, x)
        mask = repeat(mask, pattern="b c new_h new_w -> b c (new_h patch_size1) (new_w patch_size2)",
                      patch_size1=self.patch_size,
                      patch_size2=self.patch_size)
        return all_loss, mask

    def training_step(self, batch: dict, batch_index: int):
        # calculate loss
        all_loss, mask = self(batch)
        loss_mask = (mask == 1)
        masked_loss = (all_loss * loss_mask).sum() / loss_mask.sum()
        self.log(name="train_mask_loss", value=masked_loss, on_step=True, sync_dist=True)

        # use masked loss for gradient descent
        return masked_loss

    def validation_step(self, batch: dict, batch_index: int):
        all_loss, mask = self(batch)
        loss_mask = (mask == 1)
        mask_loss = (all_loss * loss_mask).sum() / loss_mask.sum()
        self.log(name="val_mask_loss", value=mask_loss, on_step=True, sync_dist=True)
        self.log(name="val_global_loss", value=all_loss.mean(), on_epoch=True, sync_dist=True)

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> Any:
        def get_output_path(tif_path_):
            tif_basename_ = os.path.splitext(os.path.basename(tif_path_))[0]
            scene_folder_ = os.path.dirname(tif_path_)
            scene_name_ = os.path.basename(scene_folder_)
            predict_folder_ = os.path.dirname(scene_folder_)
            dataset_folder_ = os.path.dirname(predict_folder_)

            output_folder_ = os.path.join(dataset_folder_, "output", scene_name_)
            os.makedirs(output_folder_, exist_ok=True)

            origin_output_path_ = os.path.join(output_folder_, f"{tif_basename_}_origin.npy")
            mask_output_path_ = os.path.join(output_folder_, f"{tif_basename_}_mask.npy")
            recover_output_path_ = os.path.join(output_folder_, f"{tif_basename_}_recover.npy")

            return origin_output_path_, mask_output_path_, recover_output_path_

        x_recover = self(batch)
        x_origin, mask, tif_path_list = batch["x"], batch["mask"], batch["tif_path"]
        mask = repeat(mask, pattern="b c new_h new_w -> b c (new_h patch_size1) (new_w patch_size2)",
                      patch_size1=self.patch_size,
                      patch_size2=self.patch_size)
        x_masked = x_origin * mask

        for i, tif_path in enumerate(tif_path_list):
            origin_output_path, mask_output_path, recover_output_path = get_output_path(tif_path)
            np.save(origin_output_path, x_origin[i, :, :, :].cpu().numpy())
            np.save(mask_output_path, x_masked[i, :, :, :].cpu().numpy())
            np.save(recover_output_path, x_recover[i, :, :, :].cpu().numpy())

        return None


class MaskGenerator:
    def __init__(self, image_size: int, channels: int, mask_patch_size: int, model_patch_size: int, mask_ratio: float):
        self.image_size = image_size
        self.channels = channels
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.image_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.scale = self.mask_patch_size // self.model_patch_size
        self.mask_patch_num_one_side = self.image_size // self.mask_patch_size
        self.model_patch_num_one_side = self.image_size // self.model_patch_size

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

    def mask_in_space_and_band(self):
        patch_num = self.channels * self.mask_patch_num_one_side ** 2
        mask_num = int(np.ceil(patch_num * self.mask_ratio))

        mask_idx = np.random.permutation(patch_num)[:mask_num]
        mask = np.zeros(patch_num, dtype=int)
        mask[mask_idx] = 1

        mask = repeat(mask, pattern="(c h w) -> c (h s1) (w s2)",
                      c=self.channels,
                      h=self.mask_patch_num_one_side,
                      w=self.mask_patch_num_one_side,
                      s1=self.scale,
                      s2=self.scale)

        return torch.Tensor(mask)

    def __call__(self, strategy: Literal["band", "space", "random"]):
        if strategy == "band":
            return self.mask_in_band()
        if strategy == "space":
            return self.mask_in_space()
        if strategy == "random":
            return self.mask_in_space_and_band()
