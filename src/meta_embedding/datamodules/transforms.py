import numpy as np
import torch
from einops import rearrange
from torchvision import transforms


class BaseTransform:
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return NotImplemented


class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """

    def __init__(self, mean: list[float], std: list[float]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        result = (image - min_value[:, None, None]) / (max_value - min_value)[:, None, None] * 255
        # torch.to_tensor will normalize again
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = rearrange(result, "c h w -> h w c")
        return result


class FMOWSentinelTrainTransform(BaseTransform):
    def __init__(self, mean: list[float], std: list[float], input_size: int):
        self.mean = mean
        self.std = std
        self.input_size = input_size

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        t = transforms.Compose([
            SentinelNormalize(self.mean, self.std),
            transforms.ToTensor(),
            transforms.RandomResizedCrop(self.input_size, scale=(0.2, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC,
                                         antialias=True),
            transforms.RandomHorizontalFlip()
        ])

        return t(image)


class FMOWSentinelEvalTransform(BaseTransform):
    def __init__(self, mean: list[float], std: list[float], input_size: int):
        self.mean = mean
        self.std = std
        self.input_size = input_size

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if self.input_size <= 224:
            # enlarge as imagenet
            temp_size = int(self.input_size / (224 / 256))
        else:
            temp_size = self.input_size

        t = transforms.Compose([
            SentinelNormalize(self.mean, self.std),
            # ToTensor's behavior is really strange
            transforms.ToTensor(),
            transforms.Resize(temp_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(self.input_size)
        ])

        return t(image)
