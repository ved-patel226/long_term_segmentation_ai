import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import torch

from config import imageDir, annFile


class COCODataset(Dataset):
    """COCO dataset wrapper that returns resized tensors for image and mask.

    Returns:
        image: torch.FloatTensor (C,H,W) in [0,1]
        mask:  torch.LongTensor (H,W) with integer class ids (0 = background)
    """

    def __init__(self, coco: COCO, image_dir, size=(256, 256), transform=None):
        self.coco = coco
        self.image_dir = image_dir
        self.transform = transform
        self.size = size
        self.img_ids = self.coco.getImgIds(catIds=[1])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.image_dir}/{img_info['file_name']}"

        image = Image.open(img_path).convert("RGB")

        orig_w, orig_h = image.size
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for i, ann in enumerate(anns, 1):
            ann_mask = self.coco.annToMask(ann).astype(np.uint8)
            mask[ann_mask == 1] = i

        if self.size is not None:
            image = image.resize(self.size, Image.BILINEAR)
            mask = Image.fromarray(mask).resize(self.size, Image.NEAREST)

        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()

        if self.transform is not None:
            image = self.transform(image)

        return img_id, mask


def main() -> None:
    from tqdm import tqdm
    import numpy as np

    coco = COCO(annFile)

    dataset = COCODataset(coco, imageDir, size=(512, 512))

    masks_mm = np.memmap(
        "all_masks.npy", dtype=np.uint8, mode="w+", shape=(len(dataset), 512, 512)
    )
    ids = []

    for i, (id, mask) in enumerate(tqdm(dataset)):
        masks_mm[i] = mask.numpy()
        ids.append(id)

    np.save("all_masks_ids.npy", np.array(ids))


if __name__ == "__main__":
    main()
