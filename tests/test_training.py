import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.model import build_model
from src.training.train import build_optimizer, build_scheduler, get_device, run_epoch
from src.utils.checkpoint import load_checkpoint, save_checkpoint


def make_dummy_loader(n=16, batch_size=4):
    images = torch.randn(n, 3, 224, 224)
    labels = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_get_device_cpu_fallback_warns_but_does_not_raise(capsys):
    device = get_device("cpu")
    assert device.type == "cpu"


def test_run_epoch_train_and_eval_modes():
    model = build_model(pretrained=False)
    loader = make_dummy_loader()
    criterion = torch.nn.CrossEntropyLoss()
    train_cfg = {"optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 0.0}
    optimizer = build_optimizer(model, train_cfg)

    train_loss, train_acc = run_epoch(model, loader, criterion, torch.device("cpu"), optimizer=optimizer)
    val_loss, val_acc = run_epoch(model, loader, criterion, torch.device("cpu"), optimizer=None)

    assert train_loss > 0
    assert 0.0 <= train_acc <= 1.0
    assert val_loss > 0
    assert 0.0 <= val_acc <= 1.0


def test_scheduler_reduces_lr_on_plateau():
    model = build_model(pretrained=False)
    train_cfg = {
        "optimizer": "adam",
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "scheduler_factor": 0.5,
        "scheduler_patience": 0,
    }
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    initial_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(1.0)  # no improvement
    scheduler.step(1.0)  # still no improvement -> should trigger reduction
    assert optimizer.param_groups[0]["lr"] < initial_lr


def test_checkpoint_round_trip(tmp_path):
    model = build_model(pretrained=False)
    optimizer = build_optimizer(model, {"optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 0.0})
    path = tmp_path / "best.pt"

    save_checkpoint(path, model=model, optimizer=optimizer, epoch=1, val_loss=0.5, val_acc=0.9, config={"seed": 42})

    new_model = build_model(pretrained=False)
    checkpoint = load_checkpoint(path, new_model, map_location="cpu")

    assert checkpoint["epoch"] == 1
    assert checkpoint["val_acc"] == 0.9
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)
