import os
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from config import imageDir, annFile, imageSize
from dataset import COCODatasetLOADER
from torch.utils.data import DataLoader
from model import SegmentationModel
from pycocotools.coco import COCO
import os
from torch.utils.data import random_split


coco = COCO(annFile)

seed_everything(42)

checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    dirpath="./checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    mode="min",
)

# logger = TensorBoardLogger("logs", name="segmentation")
logger = WandbLogger(project="segmentation-ai", log_model="all")

trainer = Trainer(
    max_epochs=-1,
    callbacks=[checkpoint_callback],
    logger=logger,
    precision="16-mixed",
    accelerator="gpu",
    log_every_n_steps=50,
    val_check_interval=0.5,
)


dataset = COCODatasetLOADER(coco, imageDir, size=imageSize)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=27,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=27,
    pin_memory=True,
    persistent_workers=True,
)


model = SegmentationModel()

logger.watch(model, log="all", log_freq=50)  # gradient and parameter logging

trainer.fit(model, train_loader)

# os.system("shutdown now")
