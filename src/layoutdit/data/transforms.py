import PIL
import torch
from torchvision.transforms import ToTensor
from transformers import AutoImageProcessor

class ComposeTransforms:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


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

class HFProcessorWithMaskRCNNBoxes:
    """
    Mask R‑CNN wants boxes in corner format.
    """
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
        HFProcessorWithMaskRCNNBoxes()
    ]
)
