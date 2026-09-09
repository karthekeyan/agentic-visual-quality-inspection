"""Inference latency/throughput benchmark for the trained casting-defect model.

This backs a GPU-validated inference-speed claim for the functional
proposal doc, so timing methodology matters:

  - GPU warm-up: the first several CUDA calls pay for context init, cuDNN
    algorithm selection/autotuning, and kernel caching -- these runs are
    discarded, not averaged in.
  - torch.cuda.synchronize() brackets every timed region. CUDA kernels are
    launched asynchronously from the host, so an un-synchronized
    time.perf_counter() around a forward pass mostly measures how fast the
    host can *enqueue* work, not how long the GPU took to run it.
  - Single-image latency is measured as its own distribution (>=100
    individual passes, batch_size=1) rather than derived by dividing a
    larger batch's wall time by N -- that would hide per-call overhead
    (kernel launch, host<->device sync) that's real for the single-image
    serving path this number is meant to represent.
  - Batch throughput (images/sec) is measured separately at a few batch
    sizes, since larger batches amortize per-call overhead differently and
    a cuDNN algorithm is autotuned per input shape.
  - Timed region is host->device transfer + forward pass only. Images are
    decoded and preprocessed ahead of time so JPEG I/O/CPU work doesn't
    contaminate the GPU inference number -- this measures model inference
    speed, not the data pipeline.
"""

import json
import statistics
import time
from pathlib import Path

import torch
import yaml

from src.data.dataset import CastingDefectDataset
from src.data.transforms import build_eval_transforms
from src.models.model import build_model
from src.utils.checkpoint import load_checkpoint

REPO = Path(__file__).resolve().parents[2]

SINGLE_IMAGE_WARMUP = 15
SINGLE_IMAGE_RUNS = 200
BATCH_SIZES = [1, 8, 32]
BATCH_WARMUP_ITERS = 10
BATCH_TIMED_ITERS = 30


def load_preprocessed_test_images(config, n):
    """Preprocess n test-set images up front (CPU tensors), so the timed
    region never includes disk I/O or JPEG decode.
    """
    data_cfg = config["data"]
    test_dir = f"{data_cfg['raw_dir']}/casting_data/casting_data/test"
    transform = build_eval_transforms(data_cfg["image_size"])
    dataset = CastingDefectDataset(test_dir, transform=transform)

    tensors = []
    i = 0
    while len(tensors) < n:
        image, _ = dataset[i % len(dataset)]
        tensors.append(image)
        i += 1
    return tensors


def benchmark_single_image(model, device, images):
    """Time SINGLE_IMAGE_RUNS individual batch_size=1 inference passes.

    Returns a list of per-call latencies in milliseconds (warm-up discarded).
    """
    all_images = images[: SINGLE_IMAGE_WARMUP + SINGLE_IMAGE_RUNS]

    latencies_ms = []
    with torch.no_grad():
        for i, image in enumerate(all_images):
            batch = image.unsqueeze(0)

            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.perf_counter()

            batch_dev = batch.to(device, non_blocking=True)
            _ = model(batch_dev)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            if i >= SINGLE_IMAGE_WARMUP:
                latencies_ms.append((end - start) * 1000.0)

    return latencies_ms


def benchmark_batch_throughput(model, device, images, batch_size):
    """Time BATCH_TIMED_ITERS forward passes at a fixed batch_size.

    Returns mean batch latency (ms) and throughput (images/sec).
    """
    # Build enough distinct batches to cycle through during warmup+timed runs
    # so we're not just re-running the exact same tensor (which wouldn't
    # change timing here, but keeps the benchmark honest/representative).
    n_batches_needed = BATCH_WARMUP_ITERS + BATCH_TIMED_ITERS
    batches = []
    for b in range(n_batches_needed):
        start_idx = (b * batch_size) % len(images)
        chunk = [images[(start_idx + j) % len(images)] for j in range(batch_size)]
        batches.append(torch.stack(chunk))

    with torch.no_grad():
        for i in range(BATCH_WARMUP_ITERS):
            batch_dev = batches[i].to(device, non_blocking=True)
            _ = model(batch_dev)
        if device.type == "cuda":
            torch.cuda.synchronize()

        batch_latencies_ms = []
        for i in range(BATCH_WARMUP_ITERS, BATCH_WARMUP_ITERS + BATCH_TIMED_ITERS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            batch_dev = batches[i].to(device, non_blocking=True)
            _ = model(batch_dev)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            batch_latencies_ms.append((end - start) * 1000.0)

    mean_batch_ms = statistics.mean(batch_latencies_ms)
    throughput_img_per_sec = batch_size / (mean_batch_ms / 1000.0)
    return mean_batch_ms, throughput_img_per_sec, batch_latencies_ms


def main():
    with open(REPO / "config/config.yaml") as f:
        config = yaml.safe_load(f)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark is meant to measure GPU inference latency; "
            "no CUDA device is available on this machine."
        )
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)

    checkpoint_path = REPO / config["model"]["checkpoint_dir"] / "best.pt"
    model = build_model(architecture="resnet18", num_classes=2, pretrained=False).to(device)
    ckpt = load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']} val_loss={ckpt['val_loss']:.4f} val_acc={ckpt['val_acc']:.4f}")
    print(f"Device: {device} ({gpu_name})\n")

    n_images_needed = max(
        SINGLE_IMAGE_WARMUP + SINGLE_IMAGE_RUNS,
        max(BATCH_SIZES) * (BATCH_WARMUP_ITERS + BATCH_TIMED_ITERS),
    )
    images = load_preprocessed_test_images(config, n_images_needed)

    # --- Single-image latency (batch_size=1, >=100 individually timed passes) ---
    print(f"Running single-image benchmark: {SINGLE_IMAGE_WARMUP} warm-up + {SINGLE_IMAGE_RUNS} timed passes...")
    latencies_ms = benchmark_single_image(model, device, images)

    single_image_stats = {
        "num_warmup_runs": SINGLE_IMAGE_WARMUP,
        "num_timed_runs": len(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms),
        "derived_throughput_img_per_sec": 1000.0 / statistics.mean(latencies_ms),
    }

    # --- Batch throughput at a few batch sizes ---
    print(f"Running batch throughput benchmark at batch sizes {BATCH_SIZES} "
          f"({BATCH_WARMUP_ITERS} warm-up + {BATCH_TIMED_ITERS} timed iters each)...\n")
    batch_results = {}
    for bs in BATCH_SIZES:
        mean_batch_ms, throughput, _ = benchmark_batch_throughput(model, device, images, bs)
        batch_results[str(bs)] = {
            "batch_size": bs,
            "num_warmup_iters": BATCH_WARMUP_ITERS,
            "num_timed_iters": BATCH_TIMED_ITERS,
            "mean_batch_latency_ms": mean_batch_ms,
            "throughput_img_per_sec": throughput,
        }

    report = {
        "gpu": gpu_name,
        "gpu_note": (
            "Titan RTX is a desktop/workstation GPU (24GB, full-size PCIe card). "
            "This benchmark reflects that hardware, not edge/embedded devices "
            "(e.g. Jetson) or mobile inference -- do not generalize this number "
            "to a deployment target without re-measuring there."
        ),
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPO)),
            "epoch": ckpt["epoch"],
            "val_loss": ckpt["val_loss"],
            "val_acc": ckpt["val_acc"],
        },
        "model": {
            "architecture": config["model"]["architecture"],
            "image_size": config["data"]["image_size"],
        },
        "methodology": {
            "timed_region": "host-to-device transfer + model forward pass only (images pre-decoded/pre-transformed)",
            "synchronization": "torch.cuda.synchronize() called immediately before and after every timed region",
            "single_image_warmup_runs": SINGLE_IMAGE_WARMUP,
        },
        "single_image_latency_ms": single_image_stats,
        "batch_throughput": batch_results,
    }

    out_path = REPO / "outputs" / "reports" / "latency_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print(f"GPU: {gpu_name} (desktop/workstation GPU -- not edge hardware)")
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")
    print("=" * 72)
    print("\nSingle-image latency (batch_size=1, "
          f"{single_image_stats['num_warmup_runs']} warm-up runs discarded, "
          f"{single_image_stats['num_timed_runs']} timed runs):")
    print(f"  {'mean':>8}: {single_image_stats['mean_ms']:.3f} ms")
    print(f"  {'median':>8}: {single_image_stats['median_ms']:.3f} ms")
    print(f"  {'min':>8}: {single_image_stats['min_ms']:.3f} ms")
    print(f"  {'max':>8}: {single_image_stats['max_ms']:.3f} ms")
    print(f"  {'std':>8}: {single_image_stats['std_ms']:.3f} ms")
    print(f"  derived throughput (1000/mean): {single_image_stats['derived_throughput_img_per_sec']:.1f} img/sec")

    print("\nBatch throughput:")
    print(f"  {'batch_size':>10} | {'mean batch latency (ms)':>26} | {'throughput (img/sec)':>20}")
    for bs in BATCH_SIZES:
        r = batch_results[str(bs)]
        print(f"  {r['batch_size']:>10} | {r['mean_batch_latency_ms']:>26.3f} | {r['throughput_img_per_sec']:>20.1f}")

    print(f"\nSaved full report to {out_path}")


if __name__ == "__main__":
    main()
