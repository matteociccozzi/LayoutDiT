from layoutdit.configuration import get_layout_dit_config
from layoutdit.evaluation.evaluator import Evaluator
from layoutdit.modeling.model import LayoutDetectionModel
from layoutdit.log import get_logger
import argparse

from layoutdit.training.trainer import Trainer

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LayoutDit training script")
    parser.add_argument(
        "--local_mode", action="store_true", help="Use local samples for training"
    )
    parser.add_argument(
        "--read_config",
        action="store_true",
        help="Read config from gs://layoutdit/layout_dit_config.json",
    )
    args = parser.parse_args()

    layout_dit_config = get_layout_dit_config()
    layout_dit_config.local_mode = args.local_mode

    logger.info("Starting LayoutDit training", extra={"supplied_args": args})

    model = LayoutDetectionModel(
        model_config=layout_dit_config.detection_model_config,
        device=layout_dit_config.train_config.device,
    )

    model = model.to(device=layout_dit_config.train_config.device)

    logger.info("Initialized model")
    trainer = Trainer(layout_dit_config, model)
    trainer.train()

    evaluator = Evaluator(model=model, layout_dit_config=layout_dit_config)
    evaluator.score()
    evaluator.visualize_preds()
    evaluator.visualize_gt()


if __name__ == "__main__":
    main()
