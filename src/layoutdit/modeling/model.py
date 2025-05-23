from typing import Optional, Callable

import fsspec
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import (
    AnchorGenerator,
)
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from layoutdit.configuration.model_config import ModelConfig
from layoutdit.log import get_logger
from layoutdit.modeling.backbone_type import BackboneType

logger = get_logger(__name__)


class LayoutDetectionModel(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        previous_layout_dit_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.fs_open: Callable = fsspec.open

        backbone, feature_map_names = self._build_backbone(
            model_config.backbone_type, device, previous_layout_dit_checkpoint
        )

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=feature_map_names,
            output_size=7,
            sampling_ratio=2,
        )

        anchor_sizes = model_config.anchor_sizes
        aspect_ratios = model_config.aspect_ratios
        anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        # FasterRCNN expects a List[Tensor[C,H,W]]
        self.model = FasterRCNN(
            backbone,
            num_classes=model_config.num_classes + 1,  # +1 for background
            rpn_anchor_generator=anchor_gen,
            box_roi_pool=box_roi_pool,
            max_size=224,
            min_size=224,
            fixed_size=(224, 224),  # fetched from dit backbone processor
            image_mean=(0.5, 0.5, 0.5),  # fetched from dit backbone processor
            image_std=(0.5, 0.5, 0.5),  # fetched from dit backbone processor
        )

    def _build_backbone(self, backbone_type, device, previous_layout_dit_checkpoint):
        if backbone_type == BackboneType.DIT:
            from layoutdit.modeling.dit_backbone import DiTWithFPN

            backbone = DiTWithFPN(pretrained=(previous_layout_dit_checkpoint is None))

            feature_map_names = ["p2", "p3", "p4", "p5", "pool"]

            if previous_layout_dit_checkpoint:
                assert device
                fs = fsspec.filesystem("gcs")
                with fs.open(previous_layout_dit_checkpoint, "rb") as f:
                    state_dict = torch.load(f, map_location=device)
                backbone.backbone.dit.load_state_dict(state_dict, strict=False)

        else:
            from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
            from torchvision.models import ResNet50_Weights

            backbone = resnet_fpn_backbone(
                backbone_name="resnet50",
                weights=ResNet50_Weights.DEFAULT,  # ImageNet pretrained
                trainable_layers=5,
                returned_layers=None,  # default = last 4 (C3–C6 → p2–p5)
                extra_blocks=LastLevelMaxPool(),  # adds a p6 via max‑pool on p5
            )

            feature_map_names = ["0", "1", "2", "3"]
        return backbone, feature_map_names

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def save_checkpoint_to_gcs(self, run_name: str, epoch_num: int):
        """
        Save two copies of this model’s weights to GCS under:
          gs://<BUCKET>/model_checkpoints/<run_name>/epoch_<epoch_num>_{gpu,cpu}.pth

        Returns:
            {
              "gpu": "gs://.../epoch_<epoch_num>_gpu.pth",
              "cpu": "gs://.../epoch_<epoch_num>_cpu.pth"
            }
        """

        fs = fsspec.filesystem("gcs")
        base_path = f"gs://layoutdit/{run_name}/model_checkpoints"
        paths = {}

        # 1) GPU checkpoint (saves with original device tensors)
        gpu_key = f"epoch_{epoch_num}_gpu.pth"
        gpu_path = f"{base_path}/{gpu_key}"
        with fs.open(gpu_path, "wb") as f:
            torch.save(self.state_dict(), f)
        paths["gpu"] = gpu_path

        # 2) CPU checkpoint (moves all tensors to CPU first)
        cpu_key = f"epoch_{epoch_num}_cpu.pth"
        cpu_path = f"{base_path}/{cpu_key}"
        cpu_state = {k: v.cpu() for k, v in self.state_dict().items()}
        with fs.open(cpu_path, "wb") as f:
            torch.save(cpu_state, f)
        paths["cpu"] = cpu_path

        return paths
