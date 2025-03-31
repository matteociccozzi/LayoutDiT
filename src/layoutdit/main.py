from pathlib import Path

from layoutdit.config import get_layout_dit_config
from layoutdit.model import LayoutDetectionModel
from layoutdit.publay_dataset import PubLayNetDataset, collate_fn
from layoutdit.train_entrypoint import train
from torch.utils.data import DataLoader
from layoutdit.log import get_logger
import argparse
import torchvision.transforms as tv_transforms
from layoutdit.transforms import ComposeTransforms

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LayoutDit training script")
    parser.add_argument(
        "--local_mode", action="store_true", help="Use local samples for training"
    )
    args = parser.parse_args()

    layout_dit_config = get_layout_dit_config()

    logger.info("Starting LayoutDit training", extra={"supplied_args": args})

    train_transforms = ComposeTransforms(
        [
            tv_transforms.Resize((800, 800)),  # resize to a fixed resolution
            tv_transforms.ToTensor(),
        ]
    )

    if args.local_mode:
        dataset = PubLayNetDataset(
            images_root_dir="../examples",
            annotations_json_path="../examples/samples.json",
            transforms=train_transforms,
        )
    else:
        data_root = Path("/home/jupyter/data")
        annotations_path = data_root / "samples.json"
        dataset = PubLayNetDataset(
            images_root_dir=str(data_root),
            annotations_json_path=str(annotations_path),
            transforms=train_transforms,
        )
        raise NotImplementedError(
            "Non-local mode not implemented yet. Use --local_mode for local training."
        )

    data_loader = DataLoader(
        dataset,
        batch_size=layout_dit_config.data_loader_config.batch_size,
        shuffle=layout_dit_config.data_loader_config.shuffle,
        num_workers=layout_dit_config.data_loader_config.num_workers,
        collate_fn=collate_fn,
    )

    model = LayoutDetectionModel()
    model = model.to(device=layout_dit_config.train_config.device)

    logger.info("Initialized model")
    train(layout_dit_config.train_config, model, data_loader)


if __name__ == "__main__":
    main()
