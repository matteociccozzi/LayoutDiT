from layoutdit.config_constructs import TrainingConfig
from layoutdit.log import get_logger
from layoutdit.train_utils import assign_targets_to_patches
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch
import torch.nn as nn
from layoutdit.model import LayoutDetectionModel
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def train(
    train_config: TrainingConfig, model: LayoutDetectionModel, data_loader: DataLoader
):
    """
    Training entrypoint for LayoutDit model
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_config.step_size,
        gamma=train_config.gamma,
    )

    # Only enable scaler for CUDA
    scaler = GradScaler(enabled=train_config.device == "cuda")

    model.train()
    for epoch in range(train_config.num_epochs):
        total_loss = torch.tensor(0.0, device=train_config.device)
        for images, targets in data_loader:
            logger.debug(f"Starting on image batch. batch_size={len(images)}")
            images = [img.to(train_config.device) for img in images]
            batch_imgs = torch.stack(images).to(train_config.device)

            optimizer.zero_grad()

            # only use autocast if device is cuda®
            if train_config.device == "cuda":
                with autocast(device_type="cuda", dtype=torch.float16):
                    class_logits, bbox_preds = model(batch_imgs)
                    loss = compute_loss(
                        class_logits,
                        bbox_preds,
                        targets,
                        batch_imgs,
                        train_config.device,
                    )
            else:
                class_logits, bbox_preds = model(batch_imgs)
                loss = compute_loss(
                    class_logits, bbox_preds, targets, batch_imgs, train_config.device
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            logger.debug(f"Finished on image batch. batch_size={len(images)}")

        scheduler.step()
        avg_loss = total_loss / len(data_loader)

        logger.info(
            f"Epoch {epoch + 1}/{train_config.num_epochs}, Loss: {avg_loss:.4f}"
        )


def compute_loss(class_logits, bbox_preds, targets, batch_imgs, device):
    """Compute the loss for the current batch"""
    loss_cls = torch.tensor(0.0, device=device)
    loss_reg = torch.tensor(0.0, device=device)

    # Compute loss per image in batch
    for i, target in enumerate(targets):
        gt_boxes = target["boxes"].to(device)
        gt_labels = target["labels"].to(device)
        # Determine patch grid size from model config (e.g., 14x14 for 224px if patch_size=16)
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
        cls_loss_i = nn.functional.cross_entropy(logits_i, tgt_classes)
        # Regression loss for positive patches
        mask = tgt_classes > 0
        if mask.any():
            preds_i = bbox_preds[i][mask]  # predicted coords for assigned patches
            target_i = tgt_boxes[mask]  # true coords
            reg_loss_i = nn.functional.smooth_l1_loss(preds_i, target_i)
        else:
            reg_loss_i = torch.tensor(0.0, device=device)
        loss_cls += cls_loss_i
        loss_reg += reg_loss_i
    # Average loss over batch
    return (loss_cls + loss_reg) / len(targets)
