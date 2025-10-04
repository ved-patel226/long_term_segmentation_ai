import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms as T


class SimpleSegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # downsample
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up1(x)
        x = self.dec1(x)
        return self.out_conv(x)


class SegmentationModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = SimpleSegNet()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _prepare_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Ensure masks are shape [B,1,H,W] and in [0,1].
        Handles masks in {0,1} or {0,255} (or other 0..255).
        """
        masks = masks.unsqueeze(1).float()  # ensure [B,1,H,W] and float
        # If masks appear to be in 0..255, normalize
        if masks.max() > 1.0:
            masks = masks / 255.0
        return masks.clamp(0.0, 1.0)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        masks = self._prepare_masks(masks)
        logits = self(imgs)
        loss = F.binary_cross_entropy_with_logits(logits, masks)
        self.log("train_loss", loss, prog_bar=True)
        _ = batch_idx  # reference to silence "not accessed"
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        masks = self._prepare_masks(masks)
        logits = self(imgs)
        loss = F.binary_cross_entropy_with_logits(logits, masks)
        self.log("val_loss", loss, prog_bar=True)
        _ = batch_idx  # reference to silence "not accessed"

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
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

    img_path = "./dataset/train2014/COCO_train2014_000000000009.jpg"

    model = SegmentationModel()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img = Image.open(img_path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize((512, 512)),
            T.ToTensor(),
        ]
    )
    img_to_tensor = transform(img).unsqueeze(0).to(device)

    summary(model, img_to_tensor.shape[1:])

    import matplotlib.pyplot as plt

    with torch.no_grad():
        output = model(img_to_tensor)
        output_img = output.squeeze().cpu().numpy()

    plt.imshow(output_img, cmap="gray")
    plt.title("Model Output")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
# ...existing code...
