import torch
from pydantic import BaseModel, Field


def get_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
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


class LayoutDitConfig(BaseModel):
    data_loader_config: DataLoaderConfig
    train_config: TrainingConfig
