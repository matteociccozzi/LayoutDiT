import random

import PIL
import torch
import torchvision.transforms.functional as F
from transformers import AutoImageProcessor


class RandomFlipTransform:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            w, h = image.size  # PIL image (width, height)
            boxes = target["boxes"]
            if boxes.numel() > 0:
                xmins = w - boxes[:, 2]
                xmaxs = w - boxes[:, 0]
                boxes[:, 0] = xmins
                boxes[:, 2] = xmaxs
                target["boxes"] = boxes
        return image, target


class RandomResizeTransform:
    def __init__(self, min_size=512, max_size=1024):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        w, h = image.size
        new_w = random.randint(self.min_size, self.max_size)
        new_h = int(h * new_w / w)
        image = image.resize((new_w, new_h))
        scale_x = new_w / w
        scale_y = new_h / h
        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target["boxes"] = boxes
        return image, target


class ResizeWithBoxes:
    def __init__(self, size):
        # size = (H, W) or int
        self.size = size

    def __call__(self, image, target):
        # image: PIL Image
        # target["boxes"]: Tensor[N,4] in [x1,y1,x2,y2], pixel coords
        orig_w, orig_h = image.size
        image = F.resize(image, self.size)
        new_h, new_w = (
            (self.size, self.size) if isinstance(self.size, int) else self.size
        )

        # compute scale factors
        scale_w = new_w / orig_w
        scale_h = new_h / orig_h

        # apply to boxes
        boxes = target["boxes"]
        # boxes is a FloatTensor[N,4]
        boxes = boxes * torch.tensor(
            [scale_w, scale_h, scale_w, scale_h], dtype=boxes.dtype
        )
        target["boxes"] = boxes
        return image, target


class ToTensorWithTarget:
    def __call__(self, image, target):
        # convert PIL→Tensor, leave target alone
        image = F.pil_to_tensor(image).float() / 255.0
        return image, target


class ComposeTransforms:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class HFProcessor:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("microsoft/dit-base")

    def __call__(self, image, target):
        # image: PIL.Image
        # target: your dict with “boxes”
        # this will:
        #  - resize to 224×224 (by default)
        #  - convert to float tensor [0,1]
        #  - normalize with DiT’s mean/std
        pixel_values = self.processor(
            images=image,  # or a list: [image]
            return_tensors="pt",
        ).pixel_values[0]  # shape [3, H, W]
        return pixel_values, target


class HFProcessorWithBoxes:
    def __init__(self, size=(224, 224)):
        self.processor = AutoImageProcessor.from_pretrained("microsoft/dit-base")
        self.size = size

    def __call__(self, image: PIL.Image.Image, target: dict):
        orig_w, orig_h = image.size

        out = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"height": self.size[0], "width": self.size[1]},
        )
        pixel_values = out.pixel_values[0]  # [3, new_h, new_w]
        new_h, new_w = pixel_values.shape[-2:]

        boxes = target["boxes"]
        scale_w = new_w / orig_w
        scale_h = new_h / orig_h
        scales = torch.tensor([scale_w, scale_h, scale_w, scale_h], device=boxes.device)
        target["boxes"] = boxes * scales

        # If you had image_id as a tensor, convert it; else leave it
        if isinstance(target.get("image_id"), torch.Tensor):
            target["image_id"] = int(target["image_id"].item())

        target["orig_size"] = (
            orig_h,
            orig_w,
        )  # keep the original size so that in eval we can rescale normalized coords
        return pixel_values, target


layout_dit_transforms = ComposeTransforms(
    [
        HFProcessorWithBoxes(),
    ]
)
