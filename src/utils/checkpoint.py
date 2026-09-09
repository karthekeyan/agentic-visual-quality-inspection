"""Checkpoint save/load helpers shared by training and inference."""

import torch


def save_checkpoint(path, model, optimizer, epoch, val_loss, val_acc, config):
    """Save model + optimizer state along with the metrics that selected it.

    `config` is embedded so a checkpoint is self-describing (architecture,
    image size, etc.) -- src.inference.predict can rebuild the right model
    without the caller having to separately track which config produced it.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, map_location=None):
    """Load a checkpoint saved by `save_checkpoint` into `model` (in place).

    Returns the full checkpoint dict (epoch, val_loss, val_acc, config) so
    callers can inspect what they loaded.
    """
    # weights_only=True: our checkpoints only ever hold tensors and plain
    # Python containers (see save_checkpoint), so this is safe and avoids
    # the arbitrary-code-execution risk of full pickle deserialization.
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
