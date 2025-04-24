from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from layoutdit.log import get_logger
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from torchvision.ops import FeaturePyramidNetwork

logger = get_logger(__name__)


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
            AutoModel.from_pretrained(hf_name, config=config)
            if pretrained
            else AutoModel.from_config(config)
        )

        d = config.num_hidden_layers
        self.layer_idxs = [d // 3, d // 2, 2 * d // 3, d]
        self.scales = [4.0, 2.0, 1.0, 0.5]
        self.hidden_size = config.hidden_size

    def forward(self, x):
        """
        Due to the GeneralizedRCNNTransform I know x will be multiple of 16 since we scale billinearly
        """
        # x: [B, 3, H, W], 3 = num channels
        B, _, H, W = x.shape
        patch_size = 16
        Gh, Gw = H // patch_size, W // patch_size

        hs = self.dit(x).hidden_states
        feats = OrderedDict()

        for i, (idx, scale) in enumerate(zip(self.layer_idxs, self.scales), start=2):
            # [B, 1+P, hidden_size] → drop CLS → [B, P, hidden_size]
            t = hs[idx][:, 1:, :]

            t = t.permute(0, 2, 1).view(B, self.hidden_size, Gh, Gw)

            if scale != 1.0:
                t = F.interpolate(
                    t, scale_factor=scale, mode="bilinear", align_corners=False
                )

            feats[f"p{i}"] = t
        return feats


class DiTWithFPN(nn.Module):
    """
    Wraps DiTBackbone + FPN (P2–P6) as one module suitable for FasterRCNN.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = DiTBackbone(pretrained=pretrained)

        # TODO DELETE ME
        for _, p in self.backbone.dit.named_parameters():
            p.requires_grad = False

        # FPN input channels are all equal to DiT hidden_size
        in_channels = [self.backbone.hidden_size] * 4
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels,
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),  # adds p6 via max-pool on p5
        )
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)  # dict of p2–p6, p6 = pool for dit backbone
        fpn_feats = self.fpn(feats)  # dict of p2–p6, p6 = pool for dit backbone
        return fpn_feats
