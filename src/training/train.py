"""Training entry point for the casting defect detection model.

    python -m src.training.train --config config/config.yaml
    python -m src.training.train --epochs 4   # quick smoke test, config unchanged
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data.dataloader import build_dataloaders
from src.models.model import build_model
from src.utils.checkpoint import save_checkpoint
from src.utils.seed import set_seed


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(requested="cuda"):
    """Resolve the training device -- never fall back to CPU silently.

    This project trains on a dedicated Titan RTX; a silent CPU fallback
    would turn a several-second epoch into a several-minute one without
    anyone noticing until the run is well underway. We still return a CPU
    device rather than raising, so the code isn't hard-blocked on a machine
    without a GPU, but the warning is impossible to miss in the log.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        print("!" * 70)
        print("WARNING: CUDA was requested but is NOT available on this machine.")
        print("Falling back to CPU -- training will be dramatically slower.")
        print("This project expects to run on a Titan RTX; if you're seeing")
        print("this on the training box, the CUDA setup is broken.")
        print("!" * 70)
        return torch.device("cpu")
    return torch.device(requested if torch.cuda.is_available() else "cpu")


def describe_device(device):
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return str(device)


def build_optimizer(model, train_cfg):
    name = train_cfg["optimizer"].lower()
    lr = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, train_cfg):
    # Reduces LR when val_loss plateaus -- simple, robust default that
    # needs no knowledge of total epoch count (unlike e.g. cosine annealing).
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg.get("scheduler_factor", 0.5),
        patience=train_cfg.get("scheduler_patience", 3),
    )


def run_epoch(model, loader, criterion, device, optimizer=None, log_interval=None):
    """Run one epoch. Trains (with backward/step) if `optimizer` is given,
    otherwise evaluates under torch.no_grad(). Returns (mean_loss, accuracy).
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, labels) in enumerate(loader):
            # Explicit host->device transfer for every batch; DataLoader
            # workers only ever produce CPU tensors.
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

            if is_train and log_interval and (batch_idx + 1) % log_interval == 0:
                print(f"    batch {batch_idx + 1:>4}/{len(loader)} | running loss: {loss.item():.4f}")

    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs from the config (e.g. for a short smoke test).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    train_cfg = config["training"]
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]

    device = get_device(train_cfg.get("device", "cuda"))
    print(f"Using device: {describe_device(device)}")

    loaders = build_dataloaders(config)
    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    checkpoint_dir = Path(config["model"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pt"

    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")
    log_interval = config["logging"].get("log_interval")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = train_cfg.get("early_stopping_patience")

    print(
        f"\nStarting training: {epochs} epochs | "
        f"train_batches={len(loaders['train'])} val_batches={len(loaders['val'])} | "
        f"device={describe_device(device)}\n"
    )
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, device, optimizer=optimizer, log_interval=log_interval
        )
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device, optimizer=None)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{current_lr:.6f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                config=config,
            )
            print(f"  -> new best val_loss={val_loss:.4f} (val_acc={val_acc:.4f}); checkpoint saved to {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping: no val_loss improvement for {early_stopping_patience} epochs.")
                break

    total_time = time.time() - start_time
    print(
        f"\nTraining finished in {total_time / 60:.1f} min ({total_time:.1f}s) "
        f"on device={describe_device(device)}"
    )
    print(f"Best checkpoint: {best_checkpoint_path} (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
