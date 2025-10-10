import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms as T

try:
    from models import UNet3M
except:
    from .models import UNet3M


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        model=None,
    ):
        super().__init__()
        self.model = model if model is not None else UNet3M()
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
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main() -> None:
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegmentationModel(model=UNet3M()).to(device)
    summary(model, (3, 512, 512), device=str(device))


if __name__ == "__main__":
    main()
