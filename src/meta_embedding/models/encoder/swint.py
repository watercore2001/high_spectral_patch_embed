import timm.models.swin_transformer

__all__ = ["SwinTransformer", "SwinBase"]

class SwinTransformer(timm.models.swin_transformer.SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, batch):
        return [super().forward(batch["x"])]


class SwinBase(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=4, window_size=8, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
                         **kwargs)
