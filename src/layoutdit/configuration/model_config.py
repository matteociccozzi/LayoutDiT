from pydantic import BaseModel

from layoutdit.modeling.backbone_type import BackboneType


class ModelConfig(BaseModel):
    backbone_type: BackboneType = BackboneType.DIT

    num_classes: int = 5

    anchor_sizes: list[tuple] = [(32,), (64,), (128,), (256,), (512,)]

    aspect_ratios: list[float] = [(0.5, 1.0, 2.0)] * 5
