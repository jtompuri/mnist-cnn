# MNIST Digit Recognition

A PyTorch CNN for MNIST handwritten digit classification, with reproducible
training, validation-based checkpointing, and test-set evaluation.

Current best result: **99.64% test accuracy** (best val_loss 0.0126 at epoch 29/30).

## Project layout

| File | Purpose |
| --- | --- |
| `model.py` | `MNISTNet` — 3 conv blocks (1→32→64→128), dropout, 2 FC layers, `log_softmax` output |
| `data.py` | Train/val/test loaders: 55k train / 5k val split (seed 42); light augmentation on train only |
| `train.py` | Training loop: Adam + cosine LR, early stopping, best-checkpoint saving, curves + CSV |
| `evaluate.py` | One-shot test evaluation; writes `confusion_matrix.png` and `misclassified.json` (or a `--out-dir`) |
| `visualize_predictions.py` | Sample prediction grids: `correct_predictions.png`, `incorrect_predictions.png` |
| `tests/test_protocol.py` | Sanity tests: shape/log_softmax, split sizes, val determinism, checkpoint loads |
| `NOTES.md` | Decision log (protocol changes, resolved issues, rejected ideas) |
| `AGENTS.md` | Conventions and rules for working in this repo |

## Setup

Python 3.12+ venv, CUDA build of PyTorch (do **not** install `+cpu` wheels):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
# development/test deps (or: .venv/bin/pip install -e ".[dev]")
.venv/bin/pip install pytest
```

MNIST data is downloaded automatically to `data/` on first run.
Training runs on CUDA when available; `train.py` prints `Using device: cuda`.

## Usage

All commands from the project root:

```bash
# Train (defaults: --epochs 30 --batch-size 64 --lr 0.0008 --seed 42 --patience 12)
.venv/bin/python train.py

# Evaluate best model once on the test set (writes confusion_matrix.png, misclassified.json)
.venv/bin/python evaluate.py
# outputs can be redirected:
.venv/bin/python evaluate.py --out-dir results/run1

# Visualize example correct/incorrect predictions
.venv/bin/python visualize_predictions.py

# Sanity tests (shape check, split sizes, validation determinism, checkpoint loads)
.venv/bin/python -m pytest tests/ -v
```

Outputs written to the project root:

- `best_model.pt` — best checkpoint by validation loss
- `training_history.csv`, `training_curves.png` — per-epoch metrics
- `confusion_matrix.png`, `misclassified.json` — test evaluation
- `correct_predictions.png`, `incorrect_predictions.png` — sample grids

## Protocol

- Split: 55,000 train / 5,000 validation (seed 42); no augmentation on validation.
- Checkpointing, early stopping, and all tuning use **validation only**.
- The test set (10,000) is evaluated exactly once, in `evaluate.py`.

## Notes

- The model outputs `log_softmax`, so the loss is `nn.NLLLoss` (not `CrossEntropyLoss`).
- Full design history and rejected alternatives (e.g. `torch.compile`, tested with
  no speedup) are in `NOTES.md`.
