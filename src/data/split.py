"""Stratified train/val split helper.

The shipped dataset only provides train/ and test/ folders, so a portion of
train/ must be held out as validation. We split per-class (stratified)
rather than a single random split so that the resulting val set preserves
the same ok/defective ratio as the full train set -- otherwise a plain
random split could easily over- or under-represent the minority class in
validation, making val metrics noisy and unreliable for model selection.
"""

import random


def stratified_split_indices(labels, val_fraction, seed):
    """Split sample indices into train/val, stratified by label.

    Args:
        labels: sequence of int class labels, one per sample (e.g.
            CastingDefectDataset.targets).
        val_fraction: fraction of each class to place in the val split.
        seed: RNG seed, for a reproducible split across runs.

    Returns:
        (train_indices, val_indices): lists of ints, disjoint, sorted.
    """
    rng = random.Random(seed)

    indices_by_class = {}
    for idx, label in enumerate(labels):
        indices_by_class.setdefault(label, []).append(idx)

    train_indices = []
    val_indices = []
    for label, indices in indices_by_class.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_val = round(len(shuffled) * val_fraction)
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices
