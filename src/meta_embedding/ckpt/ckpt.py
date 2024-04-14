import os
import shutil

import torch

from meta_embedding.models.encoder import ViTForMaeBaseDec512D1, GroupViTForMaeBaseDec512D1, MetaViTForMaeBaseDec512D1
from meta_embedding.modules import PreTrainingModule, AdamWCosineOptimArgs


def backup(folder: str):
    for filename in os.listdir(folder):
        if filename.endswith(".backup"):
            continue
        input_ckpt = os.path.join(folder, filename)
        shutil.copy2(input_ckpt, input_ckpt + ".backup")


def convert(folder: str, model: torch.nn.Module):
    for filename in os.listdir(folder):
        if filename.endswith(".backup"):
            continue
        print(filename)
        input_ckpt = os.path.join(folder, filename)
        output_ckpt = input_ckpt + ".v2"
        ckpt_dict = torch.load(input_ckpt)
        ckpt_dict["hyper_parameters"] = model.hparams
        print("save as", output_ckpt)
        torch.save(ckpt_dict, output_ckpt)
        shutil.move(output_ckpt, input_ckpt)


def vit_main():
    vit = PreTrainingModule(
        encoder=ViTForMaeBaseDec512D1(in_chans=10, patch_size=8, img_size=96, mask_ratio=0.75, global_pool="avg"),
        optim_args=AdamWCosineOptimArgs(weight_decay=0.05, warmup_epochs=5, annealing_epochs=45,
                                        max_lr=1e-4, min_lr=1e-5)
    )
    vit_folder = "/mnt/disk/xials/workspace/fmow_mae/vit_base_dec512d1/checkpoints/"

    backup(vit_folder)
    convert(vit_folder, vit)


def group_main():
    group = PreTrainingModule(
        encoder=GroupViTForMaeBaseDec512D1(in_chans=10, patch_size=8, img_size=96, mask_ratio=0.75,
                                           spatial_mask=False, global_pool="avg"),
        optim_args=AdamWCosineOptimArgs(weight_decay=0.05, warmup_epochs=5, annealing_epochs=45,
                                        max_lr=1e-4, min_lr=1e-5)
    )
    group_folder = "/mnt/disk/xials/workspace/fmow_mae/vit_group_base_dec512d1/checkpoints/"

    backup(group_folder)
    convert(group_folder, group)


def meta_main():
    meta = PreTrainingModule(
        encoder=MetaViTForMaeBaseDec512D1(in_chans=10, patch_size=8, img_size=96, mask_ratio=0.75,
                                          global_pool="avg"),
        optim_args=AdamWCosineOptimArgs(weight_decay=0.05, warmup_epochs=5, annealing_epochs=45,
                                        max_lr=1e-4, min_lr=1e-5)
    )

    meta_folder = "/mnt/disk/xials/workspace/fmow_mae/vit_meta_base_dec512d1/checkpoints/"

    backup(meta_folder)
    convert(meta_folder, meta)


if __name__ == "__main__":
    vit_main()
    group_main()
    meta_main()
