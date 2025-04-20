from typing import Optional, Callable

import torch
from pydantic import BaseModel, Field, field_serializer
from torch import nn, Tensor

from layoutdit.log import get_logger

logger = get_logger(__name__)


def get_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        # NotImplementedError: The operator 'aten::upsample_bicubic2d.out' is not currently implemented for the MPS
        # device. Since we need torch2.4 as that is what GCP VM images are booted up with, the error is solved in
        # torch2.6
        return "cpu"
    else:
        return "cpu"


class DataLoaderConfig(BaseModel):
    batch_size: int = 4
    shuffle: bool = True
    num_workers: int = 2


class TrainingConfig(BaseModel):
    device: str = Field(default_factory=get_available_device)
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    step_size: int = 10
    gamma: float = 0.1
    checkpoint_interval: int = 100

    # loss function and field serializer (needed for printing configured loss in logs)
    regression_loss_fn: Callable[[Tensor, Tensor], Tensor] = (
        nn.functional.smooth_l1_loss
    )

    @field_serializer("regression_loss_fn")
    def serialize_regression_loss_fn(self, regression_loss_fn, _info):
        return regression_loss_fn.__name__

    # loss function and field serializer (needed for printing configured loss in logs)
    class_loss_fn: Callable[[Tensor, Tensor], Tensor] = nn.functional.cross_entropy

    @field_serializer("class_loss_fn")
    def serialize_class_loss_fn(self, class_loss_fn, _info):
        return class_loss_fn.__name__


class EvalConfig(BaseModel):
    device: str = Field(default_factory=get_available_device)
    score_thresh: float = 0.0

    # if set evaluator will save predictions in a json file in same format as publaynet
    predictions_path: Optional[str] = "predictions.json"

    # visualization settings
    max_per_image: int = 10  # how many boxes to draw per image max
    visualize_dirpath_prefix: str = "visualizations"
    num_images: Optional[int] = 1  # how many images to visualize (None = all)


class LayoutDitConfig(BaseModel):
    # training config
    train_config: TrainingConfig = TrainingConfig()

    # data loader config
    data_loader_config: DataLoaderConfig = DataLoaderConfig()

    eval_config: EvalConfig = EvalConfig()

    # optional boolean flag for local mode, if true will load samples instead of train or test split
    local_mode: bool | None = None

    def __init__(self, **data):
        super().__init__(**data)
        logger.debug(
            f"Initialized LayoutDitConfig with:\n{self.model_dump_json(indent=2)}"
        )
