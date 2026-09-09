import yaml

from src.data.dataloader import build_dataloaders, compute_class_weights
from src.data.dataset import CastingDefectDataset
from src.data.split import stratified_split_indices


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def test_dataset_reads_both_classes():
    dataset = CastingDefectDataset("data/raw/casting_data/casting_data/train")
    assert len(dataset) > 0
    assert set(dataset.targets) == {0, 1}


def test_stratified_split_preserves_class_ratio():
    labels = [0] * 100 + [1] * 300
    train_idx, val_idx = stratified_split_indices(labels, val_fraction=0.2, seed=42)
    assert len(train_idx) + len(val_idx) == len(labels)
    assert set(train_idx).isdisjoint(val_idx)

    val_labels = [labels[i] for i in val_idx]
    val_ok_fraction = val_labels.count(0) / len(val_labels)
    # Should be close to the source ratio (100 / 400 = 0.25), not skewed.
    assert 0.20 <= val_ok_fraction <= 0.30


def test_compute_class_weights_favors_minority_class():
    weights = compute_class_weights([0, 0, 0, 1])  # class 0 is 3x more frequent
    assert weights[1] > weights[0]


def test_build_dataloaders_batch_shapes():
    config = load_config()
    loaders = build_dataloaders(config)

    images, labels = next(iter(loaders["train"]))
    batch_size = config["data"]["batch_size"]
    height, width = config["data"]["image_size"]
    assert images.shape == (batch_size, 3, height, width)
    assert labels.shape == (batch_size,)

    # val/test must not use the training sampler/augmentation path.
    val_images, val_labels = next(iter(loaders["val"]))
    assert val_images.shape[1:] == (3, height, width)
    assert val_labels.shape[0] == val_images.shape[0]
