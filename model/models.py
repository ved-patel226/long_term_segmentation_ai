import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as T


# from common import CBAM


# Model name is as follows:
# {net}{params in millions}M
# e.g., UNet3M means U-Net with ~3 million parameters


# V1
class UNet7M(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self._conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)  # 512 = 256 + 256 from skip

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)  # 256 = 128 + 128 from skip

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)  # 128 = 64 + 64 from skip

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)  # 64 x 512 x 512
        e2 = self.enc2(self.pool(e1))  # 128 x 256 x 256
        e3 = self.enc3(self.pool(e2))  # 256 x 128 x 128

        b = self.bottleneck(self.pool(e3))  # 512 x 64 x 64

        d3 = self.up3(b)  # 256 x 128 x 128
        d3 = torch.cat([d3, e3], dim=1)  # 512 x 128 x 128
        d3 = self.dec3(d3)  # 256 x 128 x 128

        d2 = self.up2(d3)  # 128 x 256 x 256
        d2 = torch.cat([d2, e2], dim=1)  # 256 x 256 x 256
        d2 = self.dec2(d2)  # 128 x 256 x 256

        d1 = self.up1(d2)  # 64 x 512 x 512
        d1 = torch.cat([d1, e1], dim=1)  # 128 x 512 x 512
        d1 = self.dec1(d1)  # 64 x 512 x 512

        return self.out_conv(d1)  # 1 x 512 x 512


# V2 with CBAM
class UNet3M(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self._conv_block(256, 256)

        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)  # 512 = 256 + 256 from skip
        # self.cbam_dec3 = CBAM(256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)  # 256 = 128 + 128 from skip
        # self.cbam_dec2 = CBAM(128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)  # 128 = 64 + 64 from skip
        # self.cbam_dec1 = CBAM(64)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)  # 64 x 512 x 512
        e2 = self.enc2(self.pool(e1))  # 128 x 256 x 256
        e3 = self.enc3(self.pool(e2))  # 256 x 128 x 128

        b = self.bottleneck(self.pool(e3))  # 512 x 64 x 64

        d3 = self.up3(b)  # 256 x 128 x 128
        d3 = torch.cat([d3, e3], dim=1)  # 512 x 128 x 128
        d3 = self.dec3(d3)  # 256 x 128 x 128
        # d3 = self.cbam_dec3(d3)  # 256 x 128 x 128

        d2 = self.up2(d3)  # 128 x 256 x 256
        d2 = torch.cat([d2, e2], dim=1)  # 256 x 256 x 256
        d2 = self.dec2(d2)  # 128 x 256 x 256
        # d2 = self.cbam_dec2(d2)  # 128 x 256 x 256

        d1 = self.up1(d2)  # 64 x 512 x 512
        d1 = torch.cat([d1, e1], dim=1)  # 128 x 512 x 512
        d1 = self.dec1(d1)  # 64 x 512 x 512
        # d1 = self.cbam_dec1(d1)  # 64 x 512 x 512

        return self.out_conv(d1)  # 1 x 512 x 512
