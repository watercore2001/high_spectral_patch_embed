"""
Src: https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT
"""

import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import skimage.io as io
from imgaug import augmenters as iaa
from ..transforms import SentinelNormalize
from torchvision import transforms
from einops import rearrange
from ..fmow.dataset import FMOWSentinelDataset


class SegDataset(data.Dataset):
    use_band = (1, 2, 3, 4, 5, 6, 7, 8, 11, 12)

    def __init__(self, image_root, txt_name: str = "train.txt", training=False, padding=False):
        super(SegDataset, self).__init__()
        mode = txt_name.split(".")[0]
        assert os.path.exists(image_root), "path '{}' does not exist.".format(image_root)
        image_dir = os.path.join(image_root, mode, 'img')
        mask_dir = os.path.join(image_root, mode, 'label')

        txt_path = os.path.join(image_root, "dataset", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.training = training
        self.images = [os.path.join(image_dir, x + ".tif") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".tif") for x in file_names]
        assert (len(self.images) == len(self.masks))
        # 影像预处理方法
        self.aug = iaa.Sequential([
            iaa.Rot90([0, 1, 2, 3]),
            iaa.VerticalFlip(p=0.5),
            iaa.HorizontalFlip(p=0.5),
        ])

        self.mean = [FMOWSentinelDataset.mean[i] for i in self.use_band]
        self.std = [FMOWSentinelDataset.std[i] for i in self.use_band]

        self.transform = SentinelNormalize(self.mean, self.std)

        self.padding = padding


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        img = open_image(self.images[index])

        target = np.array(Image.open(self.masks[index]).convert("P"))
        target[target == 21] = 1
        target[target == 22] = 2
        target[target == 23] = 3
        target[target == 31] = 4
        target[target == 32] = 6
        target[target == 33] = 7
        target[target == 41] = 8
        target[target == 13] = 9
        target[target == 14] = 10

        if self.training:
            img, target = self.aug(image=img, segmentation_maps=np.stack(
                (target[np.newaxis, :, :], target[np.newaxis, :, :]), axis=-1))
            target = target[0, :, :, 0]

        # padding band
        # if self.padding:
        #     b = np.mean(img, axis=2)
        #     b = np.expand_dims(b, axis=2)
        #     img = np.concatenate((img, b, b), axis=2)

        # min-max
        # kid = (img - img.min(axis=(0, 1), keepdims=True))
        # mom = (img.max(axis=(0, 1), keepdims=True) - img.min(axis=(0, 1), keepdims=True))
        # img = kid / (mom + 1e-9)
        # img, target = torch.tensor(img.copy()).permute(2, 0, 1), torch.tensor(target.copy()).long()

        # mean-std
        img = rearrange(img, 'h w c -> c h w')
        img = self.transform(img)
        target = torch.tensor(target.copy()).long()

        if self.padding:
            b = torch.mean(img, dim=0)
            b = torch.unsqueeze(b, dim=0)
            img = torch.cat((img, b, b), dim=0)

        return {"x": img, "y": target}

    def __len__(self):
        return len(self.images)


def open_image(img_path):
    img = io.imread(img_path)
    # h w c
    return img.astype(np.float32)
