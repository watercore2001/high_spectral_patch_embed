from .upernet import UPerDecoder
from torch import nn
from einops import rearrange

class UPerNetForViTBase(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(768, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, 8, 8),  # 2048, 16, 16
            nn.Dropout(0.5)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(768, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 512, 4, 4),  # 2048, 16, 16
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(768, 1024, 1, 1),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024, 1024, 2, 2),  # 2048, 16, 16
            nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(768, 2048, 1, 1),
            nn.GroupNorm(32, 2048),
            nn.GELU(),
            nn.Dropout(0.5)
            # 2048, 16, 16
        )
        self.upernet = UPerDecoder(input_dims=[256, 512, 1024, 2048], output_dim=512)
        self.num_classes = num_classes
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x
        m = [self.conv0(x), self.conv1(x), self.conv2(x), self.conv3(x)]
        x = self.upernet(m)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.fc(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x

