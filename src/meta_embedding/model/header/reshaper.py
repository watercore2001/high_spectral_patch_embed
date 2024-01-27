from torch import nn
from einops.layers.torch import Rearrange
from .base_header import BaseHeader

__all__ = ["ReshapeHeader"]


class ReshapeHeader(BaseHeader):

    def __init__(self, embedding_dim: int, num_classes: int, scale_factor: int):
        super().__init__(num_classes)
        # the first dimension is consist of batch size and num classes
        self.header = nn.Sequential(
            Rearrange(pattern="b c patch_num_in_h patch_num_in_w -> b patch_num_in_h patch_num_in_w c"),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes * scale_factor ** 2),
            # same order with patch embedding
            Rearrange(
                pattern="b patch_num_in_h patch_num_in_w (c patch_size_in_h patch_size_in_w) -> "
                "b c (patch_num_in_h patch_size_in_h) (patch_num_in_w patch_size_in_w)",
                patch_size_in_h=scale_factor, patch_size_in_w=scale_factor)
        )

    def forward(self, x):
        return self.header(x)

