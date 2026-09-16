# Agentic Visual Quality Inspection for Casting Manufacturing

A vision-based defect detection system for metal casting parts, built with PyTorch, wrapped in a six-agent LangGraph pipeline (detection, characterization, root-cause reasoning, disposition, trend tracking, and reporting), served over a FastAPI backend, and viewed through a React inspection-console dashboard.

## Documentation

See the [Functional Proposal](docs/functional-proposal.md) for the full system design, agent roles, validated results, and an honest breakdown of what's proven versus heuristic versus simulated.

## Project Structure

```
.
├── config/          # YAML/JSON configuration files (paths, hyperparameters, model settings)
├── data/            # Datasets (not tracked in git — see .gitignore)
│   ├── raw/         # Original, immutable image data
│   ├── interim/     # Intermediate data during cleaning/preprocessing
│   ├── processed/   # Final datasets ready for training
│   ├── external/    # Data from third-party sources
│   ├── knowledge_base/  # Curated defect → process-cause entries (tracked in git)
│   └── chroma_db/   # Persisted ChromaDB vector store (not tracked — regenerated via ingest)
├── models/          # Trained model artifacts
│   ├── checkpoints/ # Checkpoints saved during training
│   └── pretrained/  # Pretrained/backbone weights
├── notebooks/       # Jupyter notebooks for EDA, prototyping, and visualization
├── src/             # Source code
│   ├── data/        # Dataset classes, loaders, transforms
│   ├── models/      # Model architectures
│   ├── training/    # Training loops, optimizers, schedulers
│   ├── inference/   # Inference, Grad-CAM, latency benchmarking, model comparison
│   ├── knowledge/   # Knowledge base ingestion into ChromaDB
│   ├── agents/      # The 6-agent LangGraph pipeline + orchestrator + batch runner
│   ├── api/         # FastAPI backend exposing the pipeline over HTTP
│   └── utils/       # Shared utilities (seeding, logging, metrics, checkpointing)
├── frontend/        # React + Vite inspection-console dashboard
├── outputs/         # Generated artifacts (not tracked in git — see .gitignore)
│   ├── logs/        # Training/inference logs, inspection history
│   ├── predictions/ # Model predictions on new data
│   ├── figures/     # Plots, visualizations, Grad-CAM overlays
│   └── reports/     # Evaluation reports, model comparisons, per-inspection audit records
├── tests/           # Unit and integration tests (66 passing across ML + agents + API)
├── docs/            # Project documentation
├── requirements.txt # Python dependencies
└── .gitignore
```

## Getting Started

### 1. Environment setup

Requires **Python 3.10+** (LangGraph is incompatible with 3.8) and **Node 20+** (via [nvm](https://github.com/nvm-sh/nvm)) for the dashboard frontend.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (never committed — see `.gitignore`) with your Anthropic API key, used by the Root-Cause Agent:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 2. Data

Place raw casting images under `data/raw/` (organized by class, e.g. `data/raw/ok/` and `data/raw/defective/`). Preprocessing scripts write intermediate and final datasets to `data/interim/` and `data/processed/`.

### 3. Configuration

Model, data, training, disposition, and trend parameters are defined in `config/config.yaml`.

### 4. Training

```bash
python -m src.training.train --config config/config.yaml
```

### 5. Inference (standalone)

```bash
python -m src.inference.predict --checkpoint models/checkpoints/best.pt --input path/to/image.png
```

### 6. Knowledge base setup (required once, before running the agent pipeline)

The Root-Cause Agent retrieves from a curated defect-to-process-cause knowledge base. Embed it into ChromaDB:

```bash
python -m src.knowledge.ingest
```

### 7. Run the full agent pipeline

```bash
python -m src.agents.run_pipeline --image path/to/image.png
```

For a batch of images with a summary table:

```bash
python -m src.agents.run_batch --dir path/to/image/folder --limit 20
```

### 8. Run the dashboard

Backend (Terminal 1):

```bash
source .venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8742 --reload
```

Frontend (Terminal 2):

```bash
cd frontend
nvm use 20
npm run dev
```

Open the printed local URL (default `http://localhost:5173`) in a browser.

### 9. Tests

```bash
pytest tests/
```

For a shareable HTML report:

```bash
pip install pytest-html
pytest tests/ --html=outputs/reports/test_report.html --self-contained-html
```

## Notes

- `data/`, `models/checkpoints`, `models/pretrained`, `data/chroma_db/`, `outputs/`, `frontend/node_modules`, and `.env` are git-ignored except for `.gitkeep` placeholders and `data/knowledge_base/` — large binary artifacts, generated data, and secrets should never be committed directly.
- Notebooks in `notebooks/` are intended for exploration; reusable logic should be promoted into `src/` modules.
- Only the **Root-Cause Agent** uses a language model (Claude, via the Anthropic API); Characterization and Disposition are deterministic rule-based logic, and Trend/Reporting are statistics and formatting. See the [Functional Proposal](docs/functional-proposal.md) for the full breakdown.