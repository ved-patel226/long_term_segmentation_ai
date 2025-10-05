import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms as T


class SegmentationNet(nn.Module):
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


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
    ):
        super().__init__()
        self.model = SegmentationNet()
        self.lr = lr

        self._init_weights()

    def forward(self, x):
        return self.model(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def training_step(self, batch, _):
        imgs, masks = batch

        logits = self(imgs)
        masks = masks.unsqueeze(1)

        loss = F.binary_cross_entropy_with_logits(logits, masks.float())

        self.log("train_loss", loss, prog_bar=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)

        return loss

    def validation_step(self, batch, _):
        imgs, masks = batch

        logits = self(imgs)
        masks = masks.unsqueeze(1)

        loss = F.binary_cross_entropy_with_logits(logits, masks.float())

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
