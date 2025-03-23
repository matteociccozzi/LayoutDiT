from layoutdit.config_constructs import (
    DataLoaderConfig,
    LayoutDitConfig,
    TrainingConfig,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_data_loader_config = DataLoaderConfig()
_train_config = TrainingConfig()

layout_dit_config = LayoutDitConfig(
    data_loader_config=_data_loader_config, train_config=_train_config
)

logger.info(
    "Initialized LayoutDitConfig", extra={"config": layout_dit_config.model_dump()}
)
