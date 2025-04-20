from layoutdit.configuration.config_constructs import LayoutDitConfig, DataLoaderConfig
from layoutdit.log import get_logger
from layoutdit.loss import compute_loss
from layoutdit.data.publay_dataset import PubLayNetDataset, collate_fn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch
from layoutdit.model import LayoutDetectionModel
from torch.utils.data import DataLoader

from layoutdit.data.transforms import layout_dit_transforms

logger = get_logger(__name__)


def _build_train_dataloader(
    dataloader_config: DataLoaderConfig, local_mode: bool
) -> DataLoader:
    if local_mode:
        data_segment = "samples"
    else:
        data_segment = "train"

    dataset = PubLayNetDataset(
        images_root_dir=f"gs://layoutdit/data/{data_segment}/",
        annotations_json_path=f"gs://layoutdit/data/{data_segment}.json",
        transforms=layout_dit_transforms,
    )

    return DataLoader(
        dataset,
        batch_size=dataloader_config.batch_size,
        shuffle=dataloader_config.shuffle,
        num_workers=dataloader_config.num_workers,
        collate_fn=collate_fn,
    )


def train(layout_dit_config: LayoutDitConfig, model: LayoutDetectionModel):
    """
    Training entrypoint for LayoutDit model
    """

    dataloader = _build_train_dataloader(
        layout_dit_config.data_loader_config, layout_dit_config.local_mode
    )

    train_config = layout_dit_config.train_config

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
        for images, targets in dataloader:
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
                        regression_loss_fn=train_config.regression_loss_fn,
                        classification_loss_fn=train_config.class_loss_fn,
                    )
            else:
                class_logits, bbox_preds = model(batch_imgs)
                loss = compute_loss(
                    class_logits,
                    bbox_preds,
                    targets,
                    batch_imgs,
                    train_config.device,
                    regression_loss_fn=train_config.regression_loss_fn,
                    classification_loss_fn=train_config.class_loss_fn,
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
        avg_loss = total_loss / len(dataloader)

        logger.info(
            f"Epoch {epoch + 1}/{train_config.num_epochs}, Loss: {avg_loss:.4f}"
        )

        if (epoch + 1) % train_config.checkpoint_interval == 0:
            ckpt_path = model.save_checkpoint_to_gcs(epoch + 1)
            logger.info(f"Saved checkpoint to {ckpt_path}")
