from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import FSDPStrategy
from config import imageDir, annFile
from dataset import COCODatasetLOADER
from torch.utils.data import DataLoader
from model import SegmentationModel
from pycocotools.coco import COCO


coco = COCO(annFile)

seed_everything(42)

checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    dirpath="./checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    mode="min",
)

logger = TensorBoardLogger("logs", name="segmentation")


# strategy = FSDPStrategy(
#     cpu_offload=True,
# )


trainer = Trainer(
    max_epochs=10,
    callbacks=[checkpoint_callback],
    logger=logger,
    accelerator="cpu",
    # precision="16-mixed",
    log_every_n_steps=50,
    # strategy=strategy,
)


dataset = COCODatasetLOADER(coco, imageDir, size=(512, 512))
train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=27,
    pin_memory=True,
    persistent_workers=True,
)

model = SegmentationModel()

trainer.fit(model, train_loader)
