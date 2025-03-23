import pytest
import torch

from layoutdit.publay_dataset import PubLayNetDataset
from layoutdit.publay_dataset import collate_fn

_EXAMPLES_DIR = "examples"
_ANNOTATIONS_JSON_PATH = "examples/samples.json"

@pytest.fixture
def dataset():
    data_root = _EXAMPLES_DIR
    annotations_path = _ANNOTATIONS_JSON_PATH
    return PubLayNetDataset(data_root, annotations_path)

def test_dataset_initialization(dataset):
    assert len(dataset) > 0
    assert hasattr(dataset, 'image_info')
    assert hasattr(dataset, 'annotations')
    assert hasattr(dataset, 'cat_id_to_label')

def test_dataset_getitem(dataset):
    image, target = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3  # RGB channels
    assert image.dtype == torch.float32
    assert image.min() >= 0 and image.max() <= 1  # Normalized to [0,1]

    assert isinstance(target, dict)
    assert 'boxes' in target
    assert 'labels' in target
    assert 'image_id' in target
    assert isinstance(target['boxes'], torch.Tensor)
    assert isinstance(target['labels'], torch.Tensor)
    assert isinstance(target['image_id'], torch.Tensor)

def test_dataset_collate_fn():
    batch = [
        (torch.randn(3, 100, 100), {'boxes': torch.randn(2, 4), 'labels': torch.tensor([1, 2]), 'image_id': torch.tensor([1])}),
        (torch.randn(3, 100, 100), {'boxes': torch.randn(3, 4), 'labels': torch.tensor([1, 2, 3]), 'image_id': torch.tensor([2])})
    ]
    
    images, targets = collate_fn(batch)
    
    assert len(images) == 2
    assert len(targets) == 2