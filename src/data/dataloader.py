"""DataLoader construction for the casting defect classifier.

Class imbalance strategy
-------------------------
The train split is imbalanced (~2875 ok_front vs ~3758 def_front, i.e. ok is
the minority class -- see notebooks/01_eda.ipynb). We address this with a
`WeightedRandomSampler` on the *training* loader rather than class-weighted
loss, for a few reasons:

  1. It decouples imbalance handling from the loss function. Whatever loss
     or model src/training/train.py ends up using, it sees ~balanced
     batches and doesn't need to know about class weights at all.
  2. Balanced batches give more stable BatchNorm running statistics than
     skewed batches reweighted only after the fact via the loss -- relevant
     here since the model is a ResNet18 (build_model in src/models/model.py).
  3. It composes with future per-sample weighting. The EDA also flagged
     ~14% of images as likely blurry/over/under-exposed outliers; a later
     agent could fold a quality weight into the same sampler (multiplying
     class weight * quality weight) without touching the training loop.

A class-weighted `nn.CrossEntropyLoss(weight=...)` is a perfectly valid
alternative and is exposed here via `compute_class_weights` as a drop-in for
training loops that prefer per-loss weighting instead of per-sample
resampling -- just don't use both at once, or you'll double-correct for
the imbalance.

Validation and test loaders use no sampler and no augmentation: they must
reflect the true (imbalanced) data distribution so that metrics reported on
them are representative of real inspection performance.
"""

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.data.dataset import CastingDefectDataset
from src.data.split import stratified_split_indices
from src.data.transforms import build_eval_transforms, build_train_transforms


def compute_class_weights(targets, num_classes=2):
    """Inverse-frequency class weights, e.g. for nn.CrossEntropyLoss(weight=...).

    Alternative to the WeightedRandomSampler used by default in
    build_dataloaders -- see module docstring.
    """
    counts = torch.zeros(num_classes)
    for label in targets:
        counts[label] += 1
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights


def make_weighted_sampler(targets):
    """Per-sample WeightedRandomSampler that balances classes within each batch."""
    class_weights = compute_class_weights(targets)
    sample_weights = [class_weights[label] for label in targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(config):
    """Build train/val/test DataLoaders from the project config.

    Args:
        config: parsed config/config.yaml (see src.training.train.load_config).

    Returns:
        dict with keys "train", "val", "test" -> DataLoader.
    """
    data_cfg = config["data"]
    raw_dir = data_cfg["raw_dir"]
    image_size = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    val_fraction = data_cfg["val_split"]
    seed = config["seed"]

    # The dataset ships train/test but not val, so val is carved out of
    # train here (see src/data/split.py for why it's a stratified split).
    train_dir = f"{raw_dir}/casting_data/casting_data/train"
    test_dir = f"{raw_dir}/casting_data/casting_data/test"

    train_transform = build_train_transforms(image_size)
    eval_transform = build_eval_transforms(image_size)

    # Two dataset instances over the *same* train_dir, differing only in
    # transform. Splitting must not let val images pick up train-time
    # augmentation, and a plain Subset can't apply different transforms to
    # different indices of one underlying dataset -- so we build the file
    # list twice (cheap: just a directory scan) and select disjoint indices
    # out of each. Both scans produce identical, deterministic ordering
    # (CastingDefectDataset sorts file paths), so indices line up.
    train_dataset_full = CastingDefectDataset(train_dir, transform=train_transform)
    val_dataset_full = CastingDefectDataset(train_dir, transform=eval_transform)

    train_indices, val_indices = stratified_split_indices(
        train_dataset_full.targets, val_fraction=val_fraction, seed=seed
    )
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    test_dataset = CastingDefectDataset(test_dir, transform=eval_transform)

    train_targets = [train_dataset_full.targets[i] for i in train_indices]
    sampler = make_weighted_sampler(train_targets)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # Balanced-batch guarantee from the sampler only holds for
        # full-sized batches; drop a ragged final batch to keep every
        # training batch's class mix consistent (matters for BatchNorm).
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
