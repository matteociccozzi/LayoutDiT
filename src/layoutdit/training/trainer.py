from typing import Callable

import fsspec
from matplotlib import pyplot as plt

from layoutdit.configuration.config_constructs import LayoutDitConfig, DataLoaderConfig
from layoutdit.log import get_logger
from layoutdit.data.publay_dataset import PubLayNetDataset, collate_fn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch
from layoutdit.modeling.model import LayoutDetectionModel
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from layoutdit.training.prof_trace_handler import trace_handler

logger = get_logger(__name__)


class Trainer:
    def __init__(self, config: LayoutDitConfig, model: LayoutDetectionModel):
        self.fs_open: Callable = fsspec.open

        # history of epoch losses
        self.loss_history: list[float] = []

        self.config = config
        self.model = model.to(config.train_config.device)
        self._build_dataloader()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_scaler()

    def _build_dataloader(self):
        dl_cfg: DataLoaderConfig = self.config.data_loader_config

        segment = self.config.train_config.train_input

        dataset = PubLayNetDataset(
            images_root_dir=f"gs://layoutdit/data/{segment}/",
            annotations_json_path=f"gs://layoutdit/data/{segment}.json",
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
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                total_loss = torch.tensor(0.0, device=train_cfg.device)
                for images, targets in self.dataloader:
                    # auto vcast issue in backbone and rcnn

                    images = [img.to(train_cfg.device).half() for img in images]
                    targets = [
                        {
                            k: v.to(train_cfg.device) if torch.is_tensor(v) else v
                            for k, v in t.items()
                        }
                        for t in targets
                    ]

                    with record_function("model_forward"):
                        self.optimizer.zero_grad()
                        # forward + loss
                        if train_cfg.device == "cuda":
                            with autocast(device_type="cuda", dtype=torch.float16):
                                loss_dict = self.model(images, targets)
                        else:
                            loss_dict = self.model(images, targets)

                    # backward
                    with record_function("model_backward"):
                        loss = torch.stack(list(loss_dict.values())).sum()

                        if self.scaler.is_enabled():
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            self.optimizer.step()

                    prof.step()

                    total_loss += loss
                    logger.debug(f"Finished on image batch. batch_size={len(images)}")

                # scheduler step
                self.scheduler.step()
                avg_loss = (total_loss / len(self.dataloader)).item()
                self.loss_history.append(avg_loss)

                logger.info(
                    f"Epoch {epoch + 1}/{train_cfg.num_epochs}, Loss: {avg_loss:.4f}"
                )

                # checkpoint
                if (epoch + 1) % train_cfg.checkpoint_interval == 0:
                    ckpt_path = self.model.save_checkpoint_to_gcs(
                        self.config.run_name, epoch + 1
                    )
                    logger.info(f"Saved checkpoint to {ckpt_path}")

        self._save_loss()  # save loss to gs

    def _save_loss(self):
        fig, ax = plt.subplots()
        ax.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Loss")
        ax.set_title("Training Loss per Epoch")

        loss_path = f"gs://layoutdit/{self.config.run_name}/loss_history/loss_curve.png"

        logger.info(f"Saving loss to {loss_path}")

        with self.fs_open(loss_path, "wb") as f:
            fig.savefig(f, format="png", bbox_inches="tight")
        plt.close(fig)
