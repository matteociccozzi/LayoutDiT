import torch
from typing import Tuple


def assign_targets_to_patches(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    patch_grid_size: Tuple[int, int],
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assign each ground truth to a patch index and create target tensors.
    Args:
        gt_boxes: Tensor [N,4], gt_labels: Tensor [N]
        patch_grid_size: (patch_rows, patch_cols)
        image_size: (H, W)
    Returns:
        target_classes: Tensor [P]
        target_boxes: Tensor [P, 4]
    """
    # gt_boxes: Tensor [N,4], gt_labels: Tensor [N]
    # patch_grid_size: (patch_rows, patch_cols), image_size: (H, W)
    H, W = image_size
    patch_rows, patch_cols = patch_grid_size
    patch_h = H / patch_rows
    patch_w = W / patch_cols
    P = patch_rows * patch_cols

    target_classes = torch.zeros(P, dtype=torch.long)  # 0 = background
    target_boxes = torch.zeros((P, 4), dtype=torch.float32)
    # assign each ground truth to a patch covering its center
    for lbl, box in zip(gt_labels, gt_boxes):
        x_min, y_min, x_max, y_max = box
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        col = int(cx // patch_w)
        row = int(cy // patch_h)
        if row < patch_rows and col < patch_cols:
            patch_idx = row * patch_cols + col
            target_classes[patch_idx] = lbl  # assign the class label (non-zero)
            # assign target box (normalized coordinates [0,1] relative to image size)
            target_boxes[patch_idx] = torch.tensor(
                [x_min / W, y_min / H, x_max / W, y_max / H]
            )
    return target_classes, target_boxes
