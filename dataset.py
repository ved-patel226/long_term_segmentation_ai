from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import torch

from config import imageDir, annFile, imageSize


class COCODatasetMAKER(Dataset):
    """
    #NOTE
    COCO dataset to create all_masks_ids.npy and all_masks.npy.
    Use COCODatasetLOADER for loading the precomputed masks.
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
        mask = np.zeros((orig_h, orig_w), dtype=np.bool_)

        for i, ann in enumerate(anns, 1):
            if ann["category_id"] != 1:
                continue
            ann_mask = self.coco.annToMask(ann).astype(np.bool_)
            mask[ann_mask == 1] = 1

        if self.size is not None:
            image = image.resize(self.size, Image.BILINEAR)
            mask = Image.fromarray(mask).resize(self.size, Image.NEAREST)

        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()

        if self.transform is not None:
            image = self.transform(image)

        return img_id, mask


class COCODatasetLOADER(Dataset):
    """
    #NOTE
    COCO dataset to load precomputed all_masks_ids.npy and all_masks.npy.
    Use COCODatasetMAKER to create these files.
    """

    def __init__(self, coco: COCO, image_dir, size=(256, 256)):
        self.coco = coco
        self.image_dir = image_dir
        self.size = size

        self.mask_ids = np.load(
            "all_masks_ids.npy",
            allow_pickle=True,
        )
        self.masks = np.memmap(
            "all_masks.npy",
            dtype=np.bool_,
            mode="r",
            shape=(len(self.mask_ids), 512, 512),
        )

    def load_image(self, img_info, image_dir, size):
        with Image.open(f"{image_dir}/{img_info['file_name']}") as image:
            return TF.to_tensor(image.convert("RGB").resize(size, Image.BILINEAR))

    def __len__(self):
        return len(self.mask_ids)
        # return 100

    def __getitem__(self, idx):
        mask = torch.from_numpy(self.masks[idx].copy()).float()

        img = self.load_image(
            self.coco.loadImgs([self.mask_ids[idx]])[0], self.image_dir, self.size
        )

        return img, mask


def create_dataset() -> None:
    from tqdm import tqdm
    import numpy as np

    coco = COCO(annFile)

    dataset = COCODatasetMAKER(coco, imageDir, size=imageSize)

    masks_mm = np.memmap(
        "all_masks.npy", dtype=np.bool_, mode="w+", shape=(len(dataset), *imageSize)
    )
    ids = []

    for i, (id, mask) in enumerate(tqdm(dataset)):
        masks_mm[i] = mask.numpy()
        ids.append(id)

    np.save("all_masks_ids.npy", np.array(ids))


if __name__ == "__main__":
    create_dataset()
