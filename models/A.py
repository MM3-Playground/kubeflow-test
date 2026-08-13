from torch import nn

from .CBAM import *

class Attributor(nn.Module):
    def __init__(self, image_size):
        super(Attributor, self).__init__()

        self.mask = CBAM(inplanes=3, planes=16)

        self.m = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2), # 512 / 2 = 256
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2), # 256 / 2 = 128
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.last = nn.Linear(in_features=image_size * image_size * 2, out_features=1) # 128 * 128 * 32

    def forward(self, x):
        masked = self.mask(x) * x
        return self.last(self.m(masked))

    def get_mask(self, x):
        return (self.mask(x) + 1.0) / 2.0

    def get_masked(self, x):
        masked = self.mask(x) * x
        return masked