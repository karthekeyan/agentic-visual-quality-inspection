"""Evaluate a trained checkpoint on the held-out TEST set.

Generalized from an earlier ResNet18-only script so the exact same metric
code applies to every architecture in src/models/model.py -- necessary for
an apples-to-apples comparison across models (see
src.inference.compare_models).

Metrics: accuracy, precision/recall/F1 (positive class = defective, label
1), confusion matrix, ROC-AUC. Saves a JSON report + confusion matrix PNG.

sklearn is not installed in this environment, so metrics are computed
directly from torch tensors (ROC-AUC via the Mann-Whitney U / rank-sum
identity, which is exactly equivalent to sklearn.metrics.roc_auc_score
for the binary case).
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.data.dataloader import build_dataloaders
from src.models.model import build_model
from src.utils.checkpoint import load_checkpoint

REPO = Path(__file__).resolve().parents[2]
CLASS_NAMES = {0: "ok", 1: "defective"}


def roc_auc_binary(y_true, y_score):
    """AUC via Mann-Whitney U statistic (rank-sum). Positive class = 1."""
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(y_score)
    ranks = torch.empty_like(order, dtype=torch.float64)
    sorted_scores = y_score[order]
    n = len(y_score)
    i = 0
    rank = torch.arange(1, n + 1, dtype=torch.float64)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = rank[i:j + 1].mean()
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def evaluate(config, device):
    """Run the loaded checkpoint's model over the test set.

    Returns (report_dict, confusion_matrix_2x2_list) -- report_dict is
    JSON-serializable and matrix is [[tn, fp], [fn, tp]].
    """
    architecture = config["model"]["architecture"]
    checkpoint_path = REPO / config["model"]["checkpoint_dir"] / "best.pt"

    model = build_model(architecture=architecture, num_classes=config["model"]["num_classes"], pretrained=False).to(device)
    ckpt = load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    loaders = build_dataloaders(config)
    test_loader = loaders["test"]

    all_labels, all_preds, all_probs_defective = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_labels.append(labels)
            all_preds.append(preds.cpu())
            all_probs_defective.append(probs[:, 1].cpu())

    y_true = torch.cat(all_labels)
    y_pred = torch.cat(all_preds)
    y_score = torch.cat(all_probs_defective)

    n = len(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = roc_auc_binary(y_true, y_score)

    report = {
        "architecture": architecture,
        "checkpoint_path": str(checkpoint_path.relative_to(REPO)),
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_loss": ckpt["val_loss"],
        "checkpoint_val_acc": ckpt["val_acc"],
        "test_set_size": n,
        "test_class_counts": {"ok": int((y_true == 0).sum()), "defective": int((y_true == 1).sum())},
        "positive_class": "defective",
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
        },
        "confusion_matrix": {
            "labels": ["ok", "defective"],
            "matrix": [[tn, fp], [fn, tp]],
            "layout": "rows=true label, cols=predicted label, order=[ok, defective]",
        },
    }
    return report


def save_confusion_matrix_figure(report, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(report["confusion_matrix"]["matrix"])
    m = report["metrics"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["ok", "defective"])
    ax.set_yticklabels(["ok", "defective"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"{report['architecture']} Test Confusion Matrix\n"
        f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}"
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--out", default=None, help="Report JSON path (default: outputs/reports/test_evaluation_<architecture>.json)")
    args = parser.parse_args()

    with open(REPO / args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = evaluate(config, device)

    print(f"Loaded checkpoint from epoch {report['checkpoint_epoch']} | "
          f"val_loss={report['checkpoint_val_loss']:.4f} val_acc={report['checkpoint_val_acc']:.4f}")
    m = report["metrics"]
    print(f"\n=== TEST SET EVALUATION: {report['architecture']} (positive class = defective) ===")
    print(f"n = {report['test_set_size']} "
          f"(ok={report['test_class_counts']['ok']}, defective={report['test_class_counts']['defective']})")
    print(f"accuracy:  {m['accuracy']:.4f}")
    print(f"precision: {m['precision']:.4f}")
    print(f"recall:    {m['recall']:.4f}")
    print(f"f1:        {m['f1']:.4f}")
    print(f"roc_auc:   {m['roc_auc']:.4f}")
    cm = report["confusion_matrix"]["matrix"]
    print("\nconfusion matrix (rows=true, cols=pred), order=[ok, defective]:")
    print("           pred_ok  pred_defective")
    print(f"true_ok         {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"true_defective  {cm[1][0]:>5}          {cm[1][1]:>5}")

    out_path = REPO / (args.out or f"outputs/reports/test_evaluation_{report['architecture']}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_path}")

    cm_path = out_path.with_name(out_path.stem.replace("test_evaluation", "test_confusion_matrix") + ".png")
    save_confusion_matrix_figure(report, cm_path)
    print(f"Saved confusion matrix figure to {cm_path}")


if __name__ == "__main__":
    main()
