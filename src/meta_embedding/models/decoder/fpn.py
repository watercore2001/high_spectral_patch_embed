import torch
import torch.nn as nn
import torch.nn.functional as f


class FpnBlock(nn.Module):
    def __init__(self, input_dim: int, inner_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, inner_dim, kernel_size=1)

    def forward(self, x, lower_x=None):
        # conv + possible up sample from lower feature map
        x = self.conv(x)
        if lower_x is not None:
            x += f.interpolate(lower_x, scale_factor=2)

        return x


class ConvGnReLU(nn.Module):
    def __init__(self, inner_dim: int, out_dim: int, use_up_sample: bool):
        super().__init__()
        self.use_up_sample = use_up_sample
        self.block = nn.Sequential(nn.Conv2d(inner_dim, out_dim, (3, 3), stride=1, padding=1, bias=False),
                                   nn.GroupNorm(32, out_dim),
                                   nn.ReLU())

    def forward(self, x):
        x = self.block(x)
        if self.use_up_sample:
            x = f.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, inner_dim: int, out_dim: int, up_sample_num: int):
        super().__init__()

        blocks = [ConvGnReLU(inner_dim, out_dim, use_up_sample=bool(up_sample_num))] + \
                 [ConvGnReLU(out_dim, out_dim, use_up_sample=True) for _ in range(1, up_sample_num)]

        self.seg_block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.seg_block(x)


class FpnDecoder(nn.Module):
    """
    Feature Pyramid Networks for Object Detection CVPR 2017
    1. use fpn block to fuse pyramid feature maps
    2. use segmentation block to up sample all feature maps
    3. use concat policy to merge all feature maps
    """
    def __init__(
            self,
            input_dims: list[int],
            inner_dim: int,
            seg_dim: int,
            output_dim: int
    ):
        super().__init__()

        self.fpn_blocks = nn.ModuleList([FpnBlock(input_dim, inner_dim) for input_dim in input_dims])

        self.seg_blocks = nn.ModuleList([SegmentationBlock(inner_dim, seg_dim, up_sample_num=up_sample_num)
                                         for up_sample_num in reversed(range(0, len(input_dims)))])

        self.dropout = nn.Dropout2d(p=0.2)
        self.output_conv = nn.Sequential(
            nn.Conv2d(seg_dim*len(input_dims), output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        )

    def forward(self, feature_maps):
        # 1. use fpn block to fuse pyramid feature maps
        outputs = []
        lower_feature_map = None
        for fpn_block, feature_map in zip(reversed(self.fpn_blocks), reversed(feature_maps)):
            # deal feature map from lower to upper
            output = fpn_block(x=feature_map, lower_x=lower_feature_map)
            outputs.append(output)
            lower_feature_map = output

        # 2. use segmentation block to up sample all feature maps
        outputs = [seg_block(p) for seg_block, p in zip(self.seg_blocks, outputs)]

        # 3. use concat policy to merge all feature maps
        x = torch.cat(outputs, dim=1)
        x = self.dropout(x)
        x = self.output_conv(x)

        return x
