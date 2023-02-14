from typing import Tuple
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.optim import Adam
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import JaccardIndex


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DoubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),  # previous bias would be canceled by the batchnorm
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetMulticlassExperiment(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        feature_sizes: Union[tuple, list] = (64, 128, 256, 512),
    ):
        super(UNetMulticlassExperiment, self).__init__()
        for feature in feature_sizes:
            self.encoder.append(DoubleConvolution(in_channels, feature))
            in_channels = feature

        # UNet decoding
        for feature in reversed(feature_sizes):
            self.decoder.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )  # x2 for "skip connections"
            )
            self.decoder.append(DoubleConvolution(feature * 2, feature))

        self.bottleneck = DoubleConvolution(feature_sizes[-1], feature_sizes[-1] * 2)
        self.final_conv = nn.Conv2d(feature_sizes[0], out_channels, kernel_size=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connection_list = []

        for encode in self.encoder:
            x = encode(x)
            skip_connection_list.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connection_list = skip_connection_list[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connection_list[idx // 2]

            # resize if input % 16 != 0
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.concat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        return self.final_conv(x)

    def training_step(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        image = sample[0]
        segmentation = sample[1]
        x_est = self.forward(image)
        loss = self.loss(x_est, segmentation[0])
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss

    def validation_step(
        self, sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        image = sample[0]
        segmentation = sample[1]
        x_est = self.forward(image)
        loss = self.loss(x_est, segmentation[0])
        self.log("val_loss", loss.item(), prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        mIoU = torch.tensor([0])
        self.log("mIoU", mIoU)

    def configure_optimizers(self) -> Tuple[Optimizer, LinearLR]:
        optimizer = Adam(self.parameters(), lr=self.optimiser_params["lr"])
        scheduler = LinearLR(
            optimizer,
            start_factor=self.optimiser_params["start_factor"],
            end_factor=self.optimiser_params["end_factor"],
            total_iters=self.optimiser_params["total_iters"],
        )
        return optimizer, scheduler
