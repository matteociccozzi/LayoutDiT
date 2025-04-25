from typing import Optional

import torch
from pydantic import BaseModel, Field

from layoutdit.configuration.model_config import ModelConfig
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
    batch_size: int = 16
    shuffle: bool = True
    num_workers: int = 0


class TrainingConfig(BaseModel):
    train_input: str = "train"

    device: str = Field(default_factory=get_available_device)
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0
    step_size: int = 10
    gamma: float = 0.1
    checkpoint_interval: int = 10


class EvalConfig(BaseModel):
    eval_input: str = "val"

    device: str = Field(default_factory=get_available_device)
    score_thresh: float = 0.0

    # if set evaluator will save predictions in a json file in same format as publaynet
    eval_base_path: Optional[str] = "gs://layoutdit"

    # visualization settings
    max_per_image: int = 10  # how many boxes to draw per image max
    visualize_dirpath_prefix: str = "visualizations"
    num_images: Optional[int] = 20  # how many images to visualize (None = all)


class LayoutDitConfig(BaseModel):
    # training config
    train_config: TrainingConfig = TrainingConfig()

    # data loader config
    data_loader_config: DataLoaderConfig = DataLoaderConfig()

    eval_config: EvalConfig = EvalConfig()

    detection_model_config: ModelConfig = ModelConfig()

    run_name: str = "resnet50_samples_300"

    # optional boolean flag for local mode, if true will load samples instead of train or test split
    local_mode: bool | None = None

    def __init__(self, **data):
        super().__init__(**data)
        logger.debug(
            f"Initialized LayoutDitConfig with:\n{self.model_dump_json(indent=2)}"
        )
