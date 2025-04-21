from typing import Tuple

import torch


def compute_loss(
    class_logits,
    bbox_preds,
    targets,
    batch_imgs,
    device,
    regression_loss_fn,
    classification_loss_fn,
):
    """Compute the loss for the current batch"""
    loss_cls = torch.tensor(0.0, device=device)
    loss_reg = torch.tensor(0.0, device=device)

    # Compute loss per image in batch
    for i, target in enumerate(targets):
        gt_boxes = target["boxes"].to(device)
        gt_labels = target["labels"].to(device)

        # patch grid size from model config (e.g., 14x14 for 224px if patch_size=16)
        patch_grid = int(class_logits.shape[1] ** 0.5)  # assuming square grid
        patch_rows = patch_cols = patch_grid
        H, W = batch_imgs.shape[2], batch_imgs.shape[3]
        tgt_classes, tgt_boxes = assign_targets_to_patches(
            gt_boxes, gt_labels, (patch_rows, patch_cols), (H, W)
        )
        tgt_classes = tgt_classes.to(device)
        tgt_boxes = tgt_boxes.to(device)

        # Classification loss for this image
        logits_i = class_logits[i]  # [P, num_classes+1]
        cls_loss_i = classification_loss_fn(logits_i, tgt_classes)

        # Regression loss for positive patches
        mask = tgt_classes > 0
        if mask.any():
            preds_i = bbox_preds[i][mask]  # predicted coords for assigned patches
            target_i = tgt_boxes[mask]  # true coords
            reg_loss_i = regression_loss_fn(preds_i, target_i)
        else:
            reg_loss_i = torch.tensor(0.0, device=device)
        loss_cls += cls_loss_i
        loss_reg += reg_loss_i

    # Average loss over batch
    return (0.3*loss_cls + 0.7*loss_reg) / len(targets)


def assign_targets_to_patches(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    patch_grid_size: Tuple[int, int],
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Takes in a gt bbox and gt labels and breaks up the image into 14x14 patches (patch_grid_size).
    After this it assigns a gt box and label to a patch only if the center of the bbox lies inside the patch.
    """
    H, W = image_size
    patch_rows, patch_cols = patch_grid_size
    patch_h = H / patch_rows
    patch_w = W / patch_cols
    num_patches = (
        patch_rows * patch_cols
    )  # 196 patches in total if we have 14x14 over 224x224

    target_classes = torch.zeros(num_patches, dtype=torch.long)
    target_boxes = torch.zeros((num_patches, 4), dtype=torch.float32)

    for lbl, box in zip(gt_labels, gt_boxes):
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        # identify the box center
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        col = int(cx // patch_w)
        row = int(cy // patch_h)
        if row < patch_rows and col < patch_cols:
            idx = row * patch_cols + col
            target_classes[idx] = lbl

            # 1) center offset within patch
            dx = (cx - col * patch_w) / patch_w
            dy = (cy - row * patch_h) / patch_h

            # 2) size normalized to image dimensions, this is done so that they are always between 0,1 which is what
            # model outputs with sigmoid activation
            dw = (x2 - x1) / W
            dh = (y2 - y1) / H

            target_boxes[idx] = torch.tensor([dx, dy, dw, dh], dtype=torch.float32)
        else:
            raise ValueError("bbox center outside of all possible patches")

    return target_classes, target_boxes
