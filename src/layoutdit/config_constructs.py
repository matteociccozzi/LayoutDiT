import torch
from pydantic import BaseModel, Field

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
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    step_size: int = 10
    gamma: float = 0.1


class LayoutDitConfig(BaseModel):
    # training config
    train_config: TrainingConfig = TrainingConfig()

    # data loader config
    data_loader_config: DataLoaderConfig = DataLoaderConfig()

    # optional boolean flag for local mode, if true will load samples instead of train or test split
    local_mode: bool | None = None

    def __init__(self, **data):
        super().__init__(**data)
        logger.debug(
            f"Initialized LayoutDitConfig with:\n{self.model_dump_json(indent=2)}"
        )
