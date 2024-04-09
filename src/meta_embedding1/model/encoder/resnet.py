from torch import nn
import timm
import torch

__all__ = ["ResNet50"]

class ResNet50(nn.Module):
    def __init__(self, in_chans: int):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=False, in_chans=in_chans)
    def forward(self, batch: dict) -> list[torch.Tensor]:
        return [self.model(batch["x"])]