from torch import nn

__all__ = ['BaseHeader']


class BaseHeader(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
