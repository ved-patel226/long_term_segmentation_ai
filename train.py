import os
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from config import (
    imageDir,
    annFile,
    imageSize,
    loadPretrained,
    pretrainedPath,
    netArchitecture,
)
from dataset import COCODatasetLOADER
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import os
from torch.utils.data import random_split
from callbacks import ProgressLogger

from model.model import SegmentationModel

coco = COCO(annFile)

seed_everything(42)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    mode="min",
)

progress_logger = ProgressLogger()

# logger = TensorBoardLogger("logs", name="segmentation")
logger = WandbLogger(project="segmentation-ai", log_model="all")

trainer = Trainer(
    max_epochs=-1,
    callbacks=[checkpoint_callback, progress_logger],
    logger=logger,
    precision="bf16-mixed",
    accelerator="gpu",
    log_every_n_steps=50,
    val_check_interval=0.5,
    gradient_clip_val=1.0,
    detect_anomaly=False,  # no more nans
)


dataset = COCODatasetLOADER(coco, imageDir, size=imageSize)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

num_workers = os.cpu_count() - 1 if os.cpu_count() else 4


train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=10,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)


if loadPretrained:
    model = SegmentationModel.load_from_checkpoint(pretrainedPath)
else:
    model = SegmentationModel(netArchitecture)


logger.watch(model, log="all", log_freq=50)  # gradient and parameter logging

trainer.fit(
    model,
    train_loader,
    val_loader,
)

# os.system("shutdown now")
