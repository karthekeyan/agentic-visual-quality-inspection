"""Side-by-side comparison of trained architectures for the casting-defect task.

Assembles, per architecture:
  - test-set metrics (from src.inference.evaluate reports)
  - latency/throughput (from src.inference.benchmark_latency reports)
  - parameter count (recomputed directly from the checkpoint's state_dict,
    so it doesn't require rebuilding the model / re-downloading weights)
  - checkpoint file size on disk

into outputs/reports/model_comparison.json and a readable markdown table
at outputs/reports/model_comparison.md.

Run src.inference.evaluate and src.inference.benchmark_latency for each
config first -- this script only aggregates their JSON reports.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]

DEFAULT_CONFIGS = ["config/config.yaml", "config/config_mobilenet_v2.yaml"]


def param_count_from_checkpoint(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model_state_dict"]
    return sum(tensor.numel() for tensor in state_dict.values())


def load_json(path):
    with open(path) as f:
        return json.load(f)


def gather_model_row(config_path):
    with open(REPO / config_path) as f:
        config = yaml.safe_load(f)
    architecture = config["model"]["architecture"]
    checkpoint_path = REPO / config["model"]["checkpoint_dir"] / "best.pt"

    eval_report = load_json(REPO / f"outputs/reports/test_evaluation_{architecture}.json")
    latency_report = load_json(REPO / f"outputs/reports/latency_benchmark_{architecture}.json")

    checkpoint_size_bytes = checkpoint_path.stat().st_size
    num_params = param_count_from_checkpoint(checkpoint_path)

    m = eval_report["metrics"]
    sil = latency_report["single_image_latency_ms"]
    bt32 = latency_report["batch_throughput"]["32"]

    return {
        "architecture": architecture,
        "checkpoint_path": str(checkpoint_path.relative_to(REPO)),
        "checkpoint_epoch": eval_report["checkpoint_epoch"],
        "num_parameters": num_params,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "checkpoint_size_mb": checkpoint_size_bytes / (1024 ** 2),
        "test_metrics": {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
        },
        "confusion_matrix": eval_report["confusion_matrix"],
        "latency": {
            "single_image_mean_ms": sil["mean_ms"],
            "single_image_median_ms": sil["median_ms"],
            "single_image_min_ms": sil["min_ms"],
            "single_image_max_ms": sil["max_ms"],
            "single_image_std_ms": sil["std_ms"],
            "batch32_mean_latency_ms": bt32["mean_batch_latency_ms"],
            "batch32_throughput_img_per_sec": bt32["throughput_img_per_sec"],
        },
        "gpu": latency_report["gpu"],
    }


def build_markdown_table(rows):
    lines = [
        "| Metric | " + " | ".join(r["architecture"] for r in rows) + " |",
        "|---" * (len(rows) + 1) + "|",
    ]

    def fmt_row(label, values):
        return f"| {label} | " + " | ".join(values) + " |"

    lines.append(fmt_row("Accuracy", [f"{r['test_metrics']['accuracy']:.4f}" for r in rows]))
    lines.append(fmt_row("Precision", [f"{r['test_metrics']['precision']:.4f}" for r in rows]))
    lines.append(fmt_row("Recall", [f"{r['test_metrics']['recall']:.4f}" for r in rows]))
    lines.append(fmt_row("F1", [f"{r['test_metrics']['f1']:.4f}" for r in rows]))
    lines.append(fmt_row("ROC-AUC", [f"{r['test_metrics']['roc_auc']:.4f}" for r in rows]))
    lines.append(fmt_row("Single-image latency (mean, ms)", [f"{r['latency']['single_image_mean_ms']:.3f}" for r in rows]))
    lines.append(fmt_row("Batch-32 throughput (img/sec)", [f"{r['latency']['batch32_throughput_img_per_sec']:.1f}" for r in rows]))
    lines.append(fmt_row("Parameters", [f"{r['num_parameters']:,}" for r in rows]))
    lines.append(fmt_row("Checkpoint size (MB)", [f"{r['checkpoint_size_mb']:.2f}" for r in rows]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--out-json", default="outputs/reports/model_comparison.json")
    parser.add_argument("--out-md", default="outputs/reports/model_comparison.md")
    args = parser.parse_args()

    rows = [gather_model_row(c) for c in args.configs]

    out_json_path = REPO / args.out_json
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump({"gpu": rows[0]["gpu"], "models": rows}, f, indent=2)
    print(f"Saved {out_json_path}")

    table = build_markdown_table(rows)
    md = f"# Model Comparison: {' vs '.join(r['architecture'] for r in rows)}\n\nGPU: {rows[0]['gpu']}\n\n{table}\n"
    out_md_path = REPO / args.out_md
    with open(out_md_path, "w") as f:
        f.write(md)
    print(f"Saved {out_md_path}\n")

    print(table)


if __name__ == "__main__":
    main()
