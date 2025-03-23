from layoutdit.config import layout_dit_config
from layoutdit.model import LayoutDetectionModel
from layoutdit.publay_dataset import PubLayNetDataset, collate_fn
from layoutdit.train_entrypoint import train
from torch.utils.data import DataLoader


def main():
    dataset = PubLayNetDataset(
        images_root_dir="examples", annot_path="data/publaynet/annotations.json"
    )

    data_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn
    )

    model = LayoutDetectionModel()
    model = model.to(device=layout_dit_config.train_config.device)

    train(layout_dit_config.train_config, model, data_loader)
