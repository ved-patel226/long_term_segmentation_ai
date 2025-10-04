import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from config import imageDir, annFile
from dataset import COCODatasetLOADER
from torch.utils.data import DataLoader
from model import SegmentationModel
from pycocotools.coco import COCO
import os


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
)


dataset = COCODatasetLOADER(coco, imageDir, size=(512, 512))
train_loader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=True,
    num_workers=27,
    pin_memory=True,
    persistent_workers=True,
)

model = SegmentationModel()

logger.watch(model, log="all", log_freq=100)  # gradient and parameter logging

trainer.fit(model, train_loader)

# os.system("shutdown now")
