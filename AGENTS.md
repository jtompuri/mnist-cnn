# AGENTS.md

## Project
PyTorch CNN for MNIST digit recognition.
Files: `data.py` (loaders/splits), `model.py` (MNISTNet), `train.py` (training loop),
`evaluate.py` (test eval, confusion matrix, misclassified.json), `visualize_predictions.py` (example grids),
`pyproject.toml` (project metadata + dev extra), `conftest.py` + `tests/` (pytest suite),
`README.md` (usage), `NOTES.md` (decision log), `.github/workflows/ci.yml` (CI: pytest).

## Environment
- Python 3.14 venv in `.venv/`. Run scripts with `.venv/bin/python`.
- GPU: RTX 3090. torch is the CUDA build (torch 2.14.0+cu130).
  **Never install CPU wheels (`+cpu`)** — this happened once and silently forced training onto the CPU.
  Training must print `Using device: cuda`.
- The GPU is shared with the Ollama model serving this session; only ~2-3 GB VRAM is free. Keep batch sizes modest (default 64).

## Commands
All from the project root:
```bash
.venv/bin/python train.py                      # optional: --epochs --batch-size --lr --seed --patience
.venv/bin/python evaluate.py                   # test eval; optional --out-dir (default "."); writes confusion_matrix.png + misclassified.json
.venv/bin/python visualize_predictions.py      # writes correct/incorrect_predictions.png
.venv/bin/python -m pytest tests/              # protocol tests (shape, splits, val determinism, checkpoint) + behavior tests (loss math, seeding, checkpoint sanity)
```
`train.py` argparse (defaults in `main()`): `--epochs 30`, `--batch-size 64`, `--lr 0.0008`,
`--seed 42`, `--patience 12`. `evaluate.py` takes optional `--out-dir`. `visualize_predictions.py` takes no arguments.

## Tuning knobs
- `NUM_WORKERS` env var (default `16`): train-loader worker count in `data.py`.
  Val/test loaders stay fixed at 4. Tested on this 16-core/32-thread box: no gain above 16
  (24/32 slightly slower), see NOTES.md.

## Rules
- Never change architecture, hyperparameters or augmentation without asking first,
  and always state the change explicitly.
- Model outputs `log_softmax` -> loss is `nn.NLLLoss`, NOT `CrossEntropyLoss`.
- Convert tensors to Python scalars (`.item()` / `int()`) before printing or plotting.
- Evaluation protocol: 55 000 train / 5 000 validation split (seed 42); validation uses no augmentation.
  Checkpointing, early stopping and all tuning use **validation only**.
  The test set is evaluated exactly once, in `evaluate.py`.
- Use `tail -n 20` (or less) for logs and `head` for data files.
  Never print whole files or full training logs unless asked.
- Commit after each working change with a descriptive message.
  Before committing code changes, run `.venv/bin/python -m pytest tests/` and ensure it passes.
