# Agentic Visual Quality Inspection for Casting Manufacturing

A vision-based defect detection system for metal casting parts, built with PyTorch. The system classifies casting images (e.g. submersible pump impellers) as **ok** or **defective**, with a project structure designed to support iterating from a simple CNN classifier toward more advanced, agentic inspection pipelines.

## Project Structure

```
.
├── config/          # YAML/JSON configuration files (paths, hyperparameters, model settings)
├── data/            # Datasets (not tracked in git — see .gitignore)
│   ├── raw/         # Original, immutable image data
│   ├── interim/     # Intermediate data during cleaning/preprocessing
│   ├── processed/   # Final datasets ready for training
│   └── external/    # Data from third-party sources
├── models/          # Trained model artifacts
│   ├── checkpoints/ # Checkpoints saved during training
│   └── pretrained/  # Pretrained/backbone weights
├── notebooks/       # Jupyter notebooks for EDA, prototyping, and visualization
├── src/             # Source code
│   ├── data/        # Dataset classes, loaders, transforms
│   ├── models/      # Model architectures
│   ├── training/    # Training loops, optimizers, schedulers
│   ├── inference/   # Inference / prediction scripts
│   └── utils/       # Shared utilities (seeding, logging, metrics)
├── outputs/         # Generated artifacts (not tracked in git — see .gitignore)
│   ├── logs/        # Training/inference logs (e.g. TensorBoard)
│   ├── predictions/ # Model predictions on new data
│   ├── figures/     # Plots and visualizations
│   └── reports/     # Evaluation reports/summaries
├── tests/           # Unit and integration tests
├── docs/            # Project documentation
├── requirements.txt # Python dependencies
└── .gitignore
```

## Getting Started

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data

Place raw casting images under `data/raw/` (organized by class, e.g. `data/raw/ok/` and `data/raw/defective/`). Preprocessing scripts write intermediate and final datasets to `data/interim/` and `data/processed/`.

### 3. Configuration

Model, data, and training parameters are defined in `config/config.yaml`.

### 4. Training

```bash
python -m src.training.train --config config/config.yaml
```

### 5. Inference

```bash
python -m src.inference.predict --checkpoint models/checkpoints/best.pt --input path/to/image.png
```

### 6. Tests

```bash
pytest tests/
```

## Notes

- `data/`, `models/checkpoints`, `models/pretrained`, and `outputs/` are git-ignored except for `.gitkeep` placeholders — large binary artifacts should be stored externally (e.g. DVC, cloud storage) rather than committed directly.
- Notebooks in `notebooks/` are intended for exploration; reusable logic should be promoted into `src/` modules.
