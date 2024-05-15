from torch import nn


class LinearHeader(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.header = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.header(x)
