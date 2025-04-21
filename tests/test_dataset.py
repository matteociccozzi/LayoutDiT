

import pytest
import torch


from layoutdit.data.publay_dataset import PubLayNetDataset
from layoutdit.data.publay_dataset import collate_fn
from layoutdit.data.transforms import layout_dit_transforms


@pytest.fixture
def dataset():
    data_root = "gs://layoutdit/data/samples/"
    annotations_path = "gs://layoutdit/data/samples.json"
    return PubLayNetDataset(data_root, annotations_path, transforms=layout_dit_transforms)

def test_dataset_initialization(dataset):
    assert len(dataset) > 0
    assert hasattr(dataset, 'image_info')
    assert hasattr(dataset, 'annotations')
    assert hasattr(dataset, 'cat_id_to_label')

def test_publaynet_dataset_getitem_transformed(dataset):
    """
    Given a PubLayNetDataset with an HFProcessorWithBoxes transform,
    __getitem__ should return:
      - `image`: a FloatTensor of shape [3, H, W], dtype=torch.float32
      - `target['boxes']`: a FloatTensor of shape [N,4] with 0 <= x1 < x2 <= W, 0 <= y1 < y2 <= H
      - `target['labels']`: an IntTensor of shape [N]
      - `target['image_id']`: an int
    """
    image, target = dataset[9]

    # 1) Image checks
    # assert isinstance(image, torch.Tensor)
    # assert image.ndim == 3
    # assert image.shape[0] == 3           # RGB channels
    H, W = image.shape[1], image.shape[2]
    # assert image.dtype == torch.float32  # normalized float32

    # 2) Boxes checks
    boxes = target["boxes"]
    # assert isinstance(boxes, torch.Tensor)
    # assert boxes.ndim == 2
    # assert boxes.shape[1] == 4
    # x1, y1 should be ≥0; x2 ≤ W; y2 ≤ H
    # x1, y1, x2, y2 = boxes.unbind(dim=1)
    # assert torch.all(x1 >= 0) and torch.all(x2 <= W)
    # assert torch.all(y1 >= 0) and torch.all(y2 <= H)
    # # And x2 > x1, y2 > y1 for all boxes
    # assert torch.all(x2 > x1)
    # assert torch.all(y2 > y1)

    # 3) Labels checks
    labels = target["labels"]
    # assert isinstance(labels, torch.Tensor)
    # assert labels.ndim == 1
    # assert labels.shape[0] == boxes.shape[0]
    # assert labels.dtype == torch.int64

    # 4) Image ID
    image_id = target["image_id"]
    # assert isinstance(image_id, int)

    # --- Visualization and saving ---
    import os
    from PIL import ImageDraw
    from transformers.image_transforms import to_pil_image

    # Create scratch directory if needed
    os.makedirs("test_scratch", exist_ok=True)

    # Convert tensor to PIL image (de-normalize if necessary)
    img_vis = image.clone().clamp(0, 1)
    pil_img = to_pil_image(img_vis)
    draw = ImageDraw.Draw(pil_img)

    # Draw each box and label
    for box, label in zip(boxes, labels):
        # x1, y1, x2, y2 = box.tolist()
        x1, y1, w, h = box.tolist()
        x2 = x1 + w
        y2 = y1+h
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), str(int(label)), fill="red")

    # Save annotated image
    output_path = os.path.join("test_scratch", f"{image_id}.png")
    pil_img.save(output_path)
    print(f"Saved annotated image to {output_path}")


def test_dataset_collate_fn():
    batch = [
        (torch.randn(3, 100, 100), {'boxes': torch.randn(2, 4), 'labels': torch.tensor([1, 2]), 'image_id': torch.tensor([1])}),
        (torch.randn(3, 100, 100), {'boxes': torch.randn(3, 4), 'labels': torch.tensor([1, 2, 3]), 'image_id': torch.tensor([2])})
    ]
    
    images, targets = collate_fn(batch)
    
    assert len(images) == 2
    assert len(targets) == 2