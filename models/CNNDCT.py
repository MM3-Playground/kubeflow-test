from torch import nn

class CNNDCT(nn.Module):
    def __init__(self, image_size):
        super(CNNDCT, self).__init__()

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
        return self.last(self.m(x))