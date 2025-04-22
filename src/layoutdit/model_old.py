from typing import Optional, List, Dict

import fsspec
from torchvision.models.detection.anchor_utils import AnchorGenerator

from layoutdit.log import get_logger

logger = get_logger(__name__)

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN

class DiTBackbone(nn.Module):
    """
    Wraps a DiT model into a torchvision-style backbone returning a feature map dict.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        hf_name = "microsoft/dit-base"
        config = AutoConfig.from_pretrained(hf_name, output_hidden_states=False)
        if pretrained:
            self.dit = AutoModel.from_pretrained(hf_name, config=config)
        else:
            self.dit = AutoModel.from_config(config)
        self.out_channels = config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # runs the transformer
        outputs = self.dit(x)
        # grab last_hidden_state: [B, 1+P, C]
        feats = outputs.last_hidden_state
        B, NP, C = feats.shape
        P = NP - 1
        G = int(P ** 0.5)
        # drop CLS, reshape to [B, C, G, G]
        return feats[:, 1:, :].permute(0, 2, 1).view(B, C, G, G)

class LayoutDetectionModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        previous_layout_dit_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Implements a two-stage Mask R-CNN detector on top of DiT+FPN.
        """
        super().__init__()
        # 1) Build DiT backbone
        dit_body = DiTBackbone(pretrained=(previous_layout_dit_checkpoint is None))
        backbone = nn.Sequential(dit_body)
        # inform FPN how many channels it sees
        backbone.out_channels = dit_body.out_channels
        # optional load DiT weights manually if checkpoint provided
        if previous_layout_dit_checkpoint:
            assert device, "Please supply a device"
            fs = fsspec.filesystem("gcs")
            with fs.open(previous_layout_dit_checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=device)
            self.load_dit_state(backbone.dit, state_dict)

        # 2) Wrap with FPN
        self.backbone_with_fpn = BackboneWithFPN(
            backbone,
            return_layers={"0": "0"},
            in_channels_list=[backbone.out_channels],
            out_channels=backbone.out_channels
        )

        # 3) Build Mask R-CNN
        # num_classes+1 for background
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,)),  # one size for each of the 2 FPN outputs
            aspect_ratios=((0.5, 1.0, 2.0),) * 2  # same 3 ratios on both levels
        )
        self.model = FasterRCNN(
            self.backbone_with_fpn,
            num_classes=num_classes + 1,
            rpn_anchor_generator=anchor_generator
        )

    @staticmethod
    def load_dit_state(dit_module: nn.Module, state_dict: Dict):
        """Helper to load a raw DiT state dict into the transformer."""
        dit_module.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        On training: returns a dict of losses. On eval: returns list of detections.
        """
        return self.model(images, targets)