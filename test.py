import os
import weightwatcher as ww
from model.model import SegmentationModel

# Ensure the output directory exists
os.makedirs("ww-img", exist_ok=True)

# Load your model
model = SegmentationModel("UNet900K", ckpt_path="checkpoints/best-checkpoint-v4.ckpt")

watcher = ww.WeightWatcher(model=model)

details = watcher.analyze(plot=False, savefig="ww-img")

summary = watcher.get_summary()
details_df = watcher.get_details()

details_df.to_csv("ww-img/weightwatcher_details.csv")
