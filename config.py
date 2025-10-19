from model.common import UNet, CBAM_UNet  # Ignore

annFile = "dataset/annotations/instances_train2014.json"
imageDir = "dataset/train2014/"
# https://cocodataset.org/#download <- Download COCO dataset here

imageSize = (512, 512)  # Resize images to this size


loadPretrained = False  # Use a pretrained model
pretrainedPath = (
    "checkpoints/pretrained.ckpt"  # Path to the pretrained model checkpoint
)

MODEL_CONFIGS = {
    "UNet900K": (CBAM_UNet, [32, 64, 128, 256]),
}

netArchitecture = "UNet900K"
