
from layoutdit.config_constructs import TrainingConfig
from layoutdit.log import get_logger
import torch
from pydantic import BaseModel, Field


logger = get_logger(__name__)

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


class LayoutDitConfig(BaseModel):
    train_config: TrainingConfig = TrainingConfig()
    data_loader_config: DataLoaderConfig = DataLoaderConfig()

    def __init__(self, **data):
        super().__init__(**data)
        logger.info(f"Initialized LayoutDitConfig with:\n{self.model_dump_json(indent=2)}")


_layout_dit_config = None  
def get_layout_dit_config():
    """
    Get the LayoutDitConfig singleton.
    """
    global _layout_dit_config

    if _layout_dit_config is None:
        _layout_dit_config = LayoutDitConfig()
    
    return _layout_dit_config
