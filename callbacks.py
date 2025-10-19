import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import wandb
from config import imageSize
from pytorch_lightning.callbacks import Callback


class ProgressLogger(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            image = Image.open("image.png")
            tensor_img = (
                TF.to_tensor(image.convert("RGB").resize(imageSize, Image.BILINEAR))
                .unsqueeze(0)
                .to(pl_module.device)
            )

            pred_mask = pl_module(tensor_img)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask > 0.5).float()

        trainer.logger.experiment.log(
            {
                "epoch": trainer.current_epoch,
                "predicted_mask": wandb.Image(pred_mask[0].cpu()),
            }
        )

        pl_module.train()
