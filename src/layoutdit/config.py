from layoutdit.config_constructs import TrainingConfig, DataLoaderConfig
from layoutdit.log import get_logger
from pydantic import BaseModel


logger = get_logger(__name__)

class LayoutDitConfig(BaseModel):
    train_config: TrainingConfig = TrainingConfig()
    data_loader_config: DataLoaderConfig = DataLoaderConfig()

    def __init__(self, **data):
        super().__init__(**data)
        logger.debug(
            f"Initialized LayoutDitConfig with:\n{self.model_dump_json(indent=2)}"
        )


_layout_dit_config = None


def get_layout_dit_config():
    """
    Get the LayoutDitConfig singleton.
    """
    global _layout_dit_config

    if _layout_dit_config is None:
        _layout_dit_config = LayoutDitConfig()

    return _layout_dit_config
