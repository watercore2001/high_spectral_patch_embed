from torch import nn
from .base_header import BaseHeader

class LinearHeader(BaseHeader):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(num_classes)
        self.header = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.header(x)