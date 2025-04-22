from layoutdit.configuration.config_constructs import LayoutDitConfig, DataLoaderConfig
from layoutdit.log import get_logger
from layoutdit.data.publay_dataset import PubLayNetDataset, collate_fn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch
from layoutdit.model import LayoutDetectionModel
from torch.utils.data import DataLoader

from layoutdit.data.transforms import layout_dit_transforms

logger = get_logger(__name__)

class Trainer:
    def __init__(self, config: LayoutDitConfig, model: LayoutDetectionModel):
        self.config = config
        self.model = model.to(config.train_config.device)
        self._build_dataloader()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_scaler()

    def _build_dataloader(self):
        dl_cfg: DataLoaderConfig = self.config.data_loader_config
        segment = "samples" if self.config.local_mode else "train"
        dataset = PubLayNetDataset(
            images_root_dir=f"gs://layoutdit/data/{segment}/",
            annotations_json_path=f"gs://layoutdit/data/{segment}.json",
            transforms=layout_dit_transforms,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=dl_cfg.batch_size,
            shuffle=dl_cfg.shuffle,
            num_workers=dl_cfg.num_workers,
            collate_fn=collate_fn,
        )

    def _setup_optimizer(self):
        train_cfg = self.config.train_config
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

    def _setup_scheduler(self):
        train_cfg = self.config.train_config
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=train_cfg.step_size,
            gamma=train_cfg.gamma,
        )

    def _setup_scaler(self):
        train_cfg = self.config.train_config
        self.scaler = GradScaler(enabled=(train_cfg.device == "cuda"))

    def train(self):
        train_cfg = self.config.train_config
        self.model.train()
        for epoch in range(train_cfg.num_epochs):
            total_loss = torch.tensor(0.0, device=train_cfg.device)
            for images, targets in self.dataloader:
                images = [img.to(train_cfg.device) for img in images]
                targets = [
                    {k: v.to(train_cfg.device) if torch.is_tensor(v) else v
                     for k, v in t.items()}
                    for t in targets
                ]

                batch_imgs = torch.stack(images)

                self.optimizer.zero_grad()
                # forward + loss
                if train_cfg.device == "cuda":
                    with autocast(device_type="cuda", dtype=torch.float16):
                        loss_dict = self.model(batch_imgs, targets)
                else:
                    loss_dict = self.model(batch_imgs, targets)

                # backward
                loss = torch.stack(list(loss_dict.values())).sum()

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss
                logger.debug(f"Finished on image batch. batch_size={len(images)}")

            # scheduler step
            self.scheduler.step()
            avg_loss = total_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch+1}/{train_cfg.num_epochs}, Loss: {avg_loss:.4f}")

            # checkpoint
            if (epoch + 1) % train_cfg.checkpoint_interval == 0:
                ckpt_path = self.model.save_checkpoint_to_gcs(self.config.run_name, epoch+1)
                logger.info(f"Saved checkpoint to {ckpt_path}")
