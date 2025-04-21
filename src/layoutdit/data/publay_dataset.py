import os
import json

import fsspec
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Callable

from layoutdit.data.transforms import ComposeTransforms


class PubLayNetDataset(Dataset):
    def __init__(
        self,
        images_root_dir: str,
        annotations_json_path: str,
        transforms: ComposeTransforms = None,
    ):
        # allow seamless transition from local fs to gcfs
        self.fs_open: Callable = fsspec.open

        with self.fs_open(annotations_json_path, "r") as f:
            coco_data = json.load(f)

        # saving this so that we can use it in the Evaluator
        self.coco_data = coco_data

        self.images_root_dir = images_root_dir
        self.transforms = transforms

        self.image_info = {img["id"]: img for img in coco_data["images"]}
        self.annotations = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        self.ids = list(self.image_info.keys())
        self.cat_id_to_label = {
            cat["id"]: i for i, cat in enumerate(coco_data["categories"], start=1)
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict]:
        """
        Fetch the image and target for the given index, apply transforms if configured.
        """
        img_id = self.ids[idx]
        img_info = self.image_info[img_id]
        file_name = img_info["file_name"]
        img_path = os.path.join(self.images_root_dir, file_name)

        with self.fs_open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            # COCO format: [x, y, width, height], Mask R CNN expects  [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x+w, y+h])
            cat_id = ann["category_id"]
            labels.append(self.cat_id_to_label.get(cat_id, 0))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    """
    Collate function for the PubLayNetDataset. Returns a tuple of tensors for the batch
    """
    return tuple(zip(*batch))
