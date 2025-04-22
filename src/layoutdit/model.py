from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from layoutdit.data.transforms import gen_rcnn_transform
from layoutdit.log import get_logger

logger = get_logger(__name__)

import fsspec
from collections import OrderedDict
from typing import Optional, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import FasterRCNN

class DiTBackbone(nn.Module):
    """
    4-scale DiT feature extractor:
      - taps hidden states at layers d/3, d/2, 2d/3, d
      - upsamples/downsamples by [4×,2×,1×,½×]
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        hf_name = "microsoft/dit-base"
        config = AutoConfig.from_pretrained(hf_name, output_hidden_states=True)
        self.dit = (
            AutoModel.from_pretrained(hf_name, config=config) if pretrained
            else AutoModel.from_config(config)
        )

        d = config.num_hidden_layers
        self.layer_idxs = [d // 3, d // 2, 2 * d // 3, d]
        self.scales = [4.0, 2.0, 1.0, 0.5]
        self.hidden_size = config.hidden_size

    def forward(self, x):
        # x: [B, 3, H, W]
        B, _, H, W = x.shape

        patch_size = 16
        min_size = 224

        if H < min_size:
            pad_h = min_size - H
        else:
            pad_h = (patch_size - (H % patch_size)) % patch_size

        if W < min_size:
            pad_w = min_size - W
        else:
            pad_w = (patch_size - (W % patch_size)) % patch_size

        # F.pad takes (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
        B, _, H, W = x.shape

        Gh, Gw = H // patch_size, W // patch_size

        hs = self.dit(x).hidden_states
        feats = OrderedDict()
        _, _, C = hs[0].shape
        for i, (idx, scale) in enumerate(zip(self.layer_idxs, self.scales), start=2):
            # [B, 1+P, C] → drop CLS → [B, P, C]
            t = hs[idx][:, 1:, :]
            # → [B, C, Gh, Gw]
            t = t.permute(0, 2, 1).view(B, C, Gh, Gw)

            if scale != 1.0:
                t = F.interpolate(t, scale_factor=scale, mode="bilinear", align_corners=False)

            feats[f"p{i}"] = t
        return feats

class DiTWithFPN(nn.Module):
    """
    Wraps DiTBackbone + FPN (P2–P6) as one module suitable for FasterRCNN.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = DiTBackbone(pretrained=pretrained)
        # FPN input channels are all equal to DiT hidden_size
        in_channels = [self.backbone.hidden_size] * 4
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels,
            out_channels=256,
            extra_blocks=LastLevelMaxPool()  # adds p6 via max-pool on p5
        )
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)      # dict of p2–p5
        fpn_feats = self.fpn(feats)   # dict of p2–p6
        return fpn_feats

class LayoutDetectionModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        previous_layout_dit_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.fs_open: Callable = fsspec.open

        # 1) Build DiTWithFPN as before
        backbone = DiTWithFPN(pretrained=(previous_layout_dit_checkpoint is None))
        for param in backbone.backbone.parameters():
            param.requires_grad = False
        if previous_layout_dit_checkpoint:
            assert device
            fs = fsspec.filesystem("gcs")
            with fs.open(previous_layout_dit_checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=device)
            backbone.backbone.dit.load_state_dict(state_dict, strict=False)

        # 2) AnchorGenerator for P2–P6
        anchor_sizes = [(32,), (64,), (128,), (256,), (512,)]
        aspect_ratios = [(0.5, 1.0, 2.0)] * 5
        anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        # 3) Create a box_roi_pool that knows about p2–p5
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['p2', 'p3', 'p4', 'p5'],
            output_size=7,
            sampling_ratio=2
        )

        # 4) Plug everything into FasterRCNN
        # FasterRCNN expects a List[Tensor[C,H,W]]
        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes + 1,   # +1 for background
            rpn_anchor_generator=anchor_gen,
            box_roi_pool=box_roi_pool,
            transform=gen_rcnn_transform,
            max_size=224,
            min_size=224,
        )

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
        assert hasattr(self, "checkpoint_bucket"), "Please set self.checkpoint_bucket"
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
