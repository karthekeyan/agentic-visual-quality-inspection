"""Quick manual check that the data pipeline loads correctly.

Prints split sizes, class balance per split, and one training batch's shape
and label distribution (to confirm the WeightedRandomSampler is actually
balancing classes). Also saves a grid of augmented training samples to
outputs/figures/sample_batch.png for a visual sanity check of the
transforms (resize, flip, rotation, normalization).

Usage:
    python -m src.data.sanity_check
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from torch.utils.data import Subset

from src.data.dataloader import build_dataloaders
from src.data.transforms import denormalize

CLASS_NAMES = {0: "ok", 1: "defective"}


def get_split_targets(loader):
    """Labels for every sample in a loader's dataset (Subset-aware)."""
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        return [dataset.dataset.targets[i] for i in dataset.indices]
    return dataset.targets


def print_split_summary(name, loader):
    counts = Counter(get_split_targets(loader))
    total = sum(counts.values())
    breakdown = ", ".join(
        f"{CLASS_NAMES[label]}={count} ({count / total:.1%})"
        for label, count in sorted(counts.items())
    )
    print(f"{name:>5} split: {total:>5} images | {breakdown}")


def save_sample_grid(images, labels, out_path, n=8):
    n = min(n, images.size(0))
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))
    for i in range(n):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(CLASS_NAMES[int(labels[i])], fontsize=10)
        axes[i].axis("off")
    fig.suptitle("Sample augmented training batch")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved sample batch visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out",
        default="outputs/figures/sample_batch.png",
        help="Where to save the augmented sample grid.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    loaders = build_dataloaders(config)

    print("=== Split sizes and class balance ===")
    for name in ("train", "val", "test"):
        print_split_summary(name, loaders[name])

    print("\n=== One training batch ===")
    images, labels = next(iter(loaders["train"]))
    print(f"images: shape={tuple(images.shape)}, dtype={images.dtype}")
    print(f"labels: shape={tuple(labels.shape)}, dtype={labels.dtype}")
    batch_counts = Counter(labels.tolist())
    print(
        "label distribution in this batch: "
        + ", ".join(
            f"{CLASS_NAMES[label]}={count}" for label, count in sorted(batch_counts.items())
        )
    )

    save_sample_grid(images, labels, Path(args.out))


if __name__ == "__main__":
    main()
