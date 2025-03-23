import random
import torchvision.transforms.functional as F
import torchvision.transforms as tv_transforms


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


class ComposeTransforms:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


train_transforms = ComposeTransforms(
    [RandomResizeTransform(min_size=600, max_size=1000), RandomFlipTransform()]
)

train_transforms = ComposeTransforms([
    tv_transforms.Resize((800, 800)),  # resize to a fixed resolution
    tv_transforms.ToTensor(),
])
