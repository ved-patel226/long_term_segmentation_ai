import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms as T


class SegmentationModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        masks = masks.unsqueeze(1)  # Add channel dimension
        logits = self(imgs)
        loss = self.criterion(logits, masks.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        masks = masks.unsqueeze(1)  # Add channel dimension
        logits = self(imgs)
        loss = self.criterion(logits, masks.float())
        self.log("val_loss", loss)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


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
