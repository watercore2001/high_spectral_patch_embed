import torch.nn as nn
from torch.nn import BatchNorm2d
import torch
from .fpn import FpnDecoder


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Scene Parsing Network CVPR 2017
    """
    def __init__(self, pool_scales: list[int], input_dim: int, inner_dim: int):
        super().__init__()
        self.ppm = nn.ModuleList([nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=scale),
                nn.Conv2d(input_dim, inner_dim, kernel_size=1, bias=False),
                BatchNorm2d(inner_dim),
                nn.ReLU()
            ) for scale in pool_scales])

        output_dim = input_dim

        self.output_conv = nn.Sequential(
            nn.Conv2d(input_dim + len(pool_scales) * inner_dim, output_dim,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: last feature map
        """
        _, _, h, w = x.size()

        ppm_outs = [x] + [nn.functional.interpolate(input=pool(x), size=(h, w), mode='bilinear') for pool in self.ppm]
        ppm_outs = torch.cat(ppm_outs, dim=1)
        return self.output_conv(ppm_outs)


class UPerDecoder(nn.Module):
    def __init__(self, input_dims: list[int], output_dim: int = 512, use_ppm=False,
                 pool_scales: list[int] = (1, 2, 3, 6)):
        super().__init__()

        self.use_ppm = use_ppm
        if self.use_ppm:
            ppm_dim = input_dims[-1]
            self.ppm = PyramidPoolingModule(pool_scales, input_dim=ppm_dim, inner_dim=ppm_dim)

        # because will concat four seg_dim, set seg_dim = 0.25 inner_dim
        self.fpn = FpnDecoder(input_dims, inner_dim=output_dim, seg_dim=output_dim//4, output_dim=output_dim)

    def forward(self, x):
        # use ppm for last feature map
        if self.use_ppm:
            x = [*x[:-1], self.ppm(x[-1])]
        x = self.fpn(x)

        return x
