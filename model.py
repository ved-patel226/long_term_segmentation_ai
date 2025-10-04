import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms as T


# Basic U-Net block
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_classes=90):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        bn = self.bottleneck(self.pool3(d3))

        u3 = self.up3(bn)
        u3 = self.dec3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3)
        u2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, d1], dim=1))

        return self.out_conv(u1)  # Shape: [B, 90, H, W]


# Lightning module
class SegmentationModel(pl.LightningModule):
    def __init__(self, n_classes=90, lr=1e-3):
        super().__init__()
        self.model = UNet(n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch  # imgs: [B,3,H,W], masks: [B,H,W] with class indices
        logits = self(imgs)
        loss = self.criterion(logits, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.criterion(logits, masks)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main() -> None:
    from torchsummary import summary

    img_path = "./dataset/train2014/COCO_train2014_000000000009.jpg"

    model = SegmentationModel(n_classes=90)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img = Image.open(img_path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
        ]
    )
    img_to_tensor = transform(img).unsqueeze(0).to(device)

    summary(model, img_to_tensor.shape[1:])


if __name__ == "__main__":
    main()
