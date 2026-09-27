# MNIST Digit Recognition

A PyTorch CNN for MNIST handwritten digit classification, with reproducible
training, validation-based checkpointing, and test-set evaluation.

Current best result: **99.64% test accuracy** (best val_loss 0.0141 at epoch 27/30).

## Project layout

| File | Purpose |
| --- | --- |
| `model.py` | `MNISTNet` — 3 conv blocks (1→32→64→128), dropout, 2 FC layers, `log_softmax` output |
| `data.py` | Train/val/test loaders: 55k train / 5k val split (seed 42); light augmentation on train only |
| `train.py` | Training loop: Adam + cosine LR, early stopping, best-checkpoint saving, curves + CSV |
| `evaluate.py` | One-shot test evaluation; writes `confusion_matrix.png` and `misclassified.json` (or a `--out-dir`) |
| `visualize_predictions.py` | Sample prediction grids: `correct_predictions.png`, `incorrect_predictions.png` (optional `--out-dir`, `--num-examples`) |
| `visualize_features.py` | Learned-feature grids: `conv1_filters.png` (32 conv1 kernels) + `feature_maps.png` (per-digit block activation maps) |
| `utils.py` | Shared `get_device` (CUDA -> MPS -> CPU), `set_seed`, `load_model` |
| `Makefile` | Make targets: `setup train eval viz features test lint format typecheck check` |
| `pyproject.toml` | Package metadata, dependencies, dev extra (pytest, ruff, pyright), tool config |
| `tests/test_protocol.py` | Sanity tests: shape/log_softmax, split sizes, val determinism, checkpoint loads |
| `tests/test_evaluation.py` | Behavior tests: loss weighting + accuracy math, seed reproducibility, checkpoint-is-trained |
| `NOTES.md` | Decision log (protocol changes, resolved issues, rejected ideas) |
| `AGENTS.md` | Conventions and rules for working in this repo |
| `.github/workflows/ci.yml` | GitHub CI: ruff + pyright + pytest on push/PR |

## Setup

Python 3.12+ venv, CUDA build of PyTorch (do **not** install `+cpu` wheels):

```bash
python -m venv .venv
make setup          # = .venv/bin/pip install -e ".[dev]"
```

MNIST data is downloaded automatically to `data/` on first run.
Training runs on CUDA when available, falling back to MPS (Apple Silicon) and then CPU;
`train.py` prints the chosen device (e.g. `Using device: cuda`).
Train-loader worker count is tunable via the `NUM_WORKERS` env var (default 16).

## Usage

All commands from the project root — either the Makefile or the raw scripts:

```bash
make train          # = .venv/bin/python train.py
make eval           # = .venv/bin/python evaluate.py
make viz            # = .venv/bin/python visualize_predictions.py
make test           # = .venv/bin/python -m pytest tests/
make check          # lint + typecheck + test

# Train (defaults: --epochs 30 --batch-size 64 --lr 0.0008 --seed 42 --patience 12)
.venv/bin/python train.py

# Evaluate best model once on the test set (writes confusion_matrix.png, misclassified.json)
.venv/bin/python evaluate.py
# outputs can be redirected:
.venv/bin/python evaluate.py --out-dir results/run1

# Visualize example correct/incorrect predictions
.venv/bin/python visualize_predictions.py
# outputs can be redirected and sized:
.venv/bin/python visualize_predictions.py --out-dir results/run1 --num-examples 20 --cols 5

# Visualize learned features: conv1 kernel grid + per-digit conv-block activation maps
# (loads best_model.pt; writes conv1_filters.png and feature_maps.png)
.venv/bin/python visualize_features.py
# outputs can be redirected:
.venv/bin/python visualize_features.py --out-dir results/run1 --model-path best_model.pt

# Sanity tests (shape check, split sizes, validation determinism, checkpoint loads)
.venv/bin/python -m pytest tests/ -v

# Lint / typecheck / format
make lint format typecheck
```

Outputs written to `results/` (override per-run with `--out-dir`):

- `best_model.pt` — best checkpoint by validation loss (project root)
- `results/training_history.csv`, `results/training_curves.png` — per-epoch metrics
- `results/confusion_matrix.png`, `results/misclassified.json` — test evaluation
- `results/correct_predictions.png`, `results/incorrect_predictions.png` — sample grids
- `results/conv1_filters.png`, `results/feature_maps.png` — learned feature visualizations

## Protocol

- Split: 55,000 train / 5,000 validation (seed 42); no augmentation on validation.
- Checkpointing, early stopping, and all tuning use **validation only**.
- The test set (10,000) is evaluated exactly once, in `evaluate.py`.

## Notes

- The model outputs `log_softmax`, so the loss is `nn.NLLLoss` (not `CrossEntropyLoss`).
- `best_model.pt` is not bit-reproducible: augmentation RNG depends on per-run worker/
  thread interleaving, so identical-config reruns land at test 99.63–99.64%.
- Full design history and rejected alternatives (e.g. `torch.compile` with no speedup,
  GPU-side augmentation with a quality cost, loader workers above 16 with no gain) are
  in `NOTES.md`.
