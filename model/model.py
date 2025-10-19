import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as T

try:
    from models import build_unet
    from common import dice_loss
except:
    from .models import build_unet
    from .common import dice_loss


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model="UNet483K",
        lr=1e-3,
        ckpt_path=None,  # new
    ):
        super().__init__()
        self.model = build_unet(model)
        self.lr = lr

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cuda")
            self.load_state_dict(ckpt["state_dict"], strict=True)
        else:
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
        logits = torch.clamp(logits, -10, 10)

        masks = masks.unsqueeze(1).float()

        dice_loss_value = dice_loss(logits, masks)
        self.log("train_dice_loss", dice_loss_value)

        binary_loss_value = F.binary_cross_entropy_with_logits(logits, masks)
        self.log("train_bce_loss", binary_loss_value)

        loss = binary_loss_value + dice_loss_value

        self.log("train_loss", loss, prog_bar=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)

        return loss

    def validation_step(self, batch, _):
        imgs, masks = batch

        logits = self(imgs)
        logits = torch.clamp(logits, -10, 10)

        masks = masks.unsqueeze(1)

        dice_loss_value = dice_loss(logits, masks)
        self.log("val_dice_loss", dice_loss_value)

        binary_loss_value = F.binary_cross_entropy_with_logits(logits, masks)
        self.log("val_bce_loss", binary_loss_value)

        loss = binary_loss_value + dice_loss_value

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
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
    model = SegmentationModel(model=build_unet("UNet483K")).to(device)
    summary(model, (3, 512, 512), device=str(device))


if __name__ == "__main__":
    main()
