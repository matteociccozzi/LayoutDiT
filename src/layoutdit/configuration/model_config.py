from pydantic import BaseModel

from layoutdit.modeling.backbone_type import BackboneType


class ModelConfig(BaseModel):
    backbone_type: BackboneType = BackboneType.DIT

    num_classes: int = 5
