import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DoubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),   # previous bias would be canceled by the batchnorm
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_sizes=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # U-Net encoding
        for feature in feature_sizes:
            self.encoder.append(DoubleConvolution(in_channels, feature))
            in_channels = feature

        # U-Net decoding
        for feature in reversed(feature_sizes):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)     # x2 for "skip connections"
            )
            self.decoder.append(DoubleConvolution(feature*2, feature))

        self.bottleneck = DoubleConvolution(feature_sizes[-1], feature_sizes[-1]*2)
        self.final_conv = nn.Conv2d(feature_sizes[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connection_list = []

        for encode in self.encoder:
            x = encode(x)
            skip_connection_list.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connection_list = skip_connection_list[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connection_list[idx//2]

            # resize if input % 16 != 0
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.concat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNet(in_channels=1, out_channels=1)
    prediction = model(x)
    print(prediction.shape, x.shape)
    assert prediction.shape == x.shape


if __name__ == "__main__":
    test()
