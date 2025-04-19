from layoutdit.config import get_layout_dit_config
from layoutdit.model import LayoutDetectionModel
from layoutdit.train_entrypoint import train
from layoutdit.log import get_logger
import argparse

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LayoutDit training script")
    parser.add_argument(
        "--local_mode", action="store_true", help="Use local samples for training"
    )
    args = parser.parse_args()

    layout_dit_config = get_layout_dit_config()
    layout_dit_config.local_mode = args.local_mode

    logger.info("Starting LayoutDit training", extra={"supplied_args": args})

    model = LayoutDetectionModel()
    model = model.to(device=layout_dit_config.train_config.device)

    logger.info("Initialized model")
    train(layout_dit_config, model)


if __name__ == "__main__":
    main()
