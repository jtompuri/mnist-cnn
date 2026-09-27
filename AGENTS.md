# AGENTS.md

## Project
PyTorch CNN for MNIST digit recognition.
Files: `data.py` (loaders/splits), `model.py` (MNISTNet), `train.py` (training loop),
`evaluate.py` (test eval, confusion matrix, misclassified.json), `visualize_predictions.py` (example grids),
`utils.py` (get_device CUDA->MPS->CPU, set_seed, load_model), `Makefile` (setup/train/eval/viz/test/lint/typecheck/check),
`pyproject.toml` (project metadata + dev extra + ruff/pyright config), `conftest.py` + `tests/` (pytest suite),
`README.md` (usage), `NOTES.md` (decision log), `.github/workflows/ci.yml` (CI: ruff + pyright + pytest).

## Environment
- Python 3.14 venv in `.venv/`. Run scripts with `.venv/bin/python`.
- GPU: RTX 3090. torch is the CUDA build (torch 2.14.0+cu130).
  **Never install CPU wheels (`+cpu`)** — this happened once and silently forced training onto the CPU.
  Training must print `Using device: cuda`.
- The GPU is shared with the Ollama model serving this session; only ~2-3 GB VRAM is free. Keep batch sizes modest (default 64).

## Commands
All from the project root (or via the Makefile: `make setup train eval viz test lint format typecheck check`):
```bash
.venv/bin/python train.py                      # optional: --epochs --batch-size --lr --seed --patience
.venv/bin/python evaluate.py                   # test eval; optional --out-dir (default "."); writes confusion_matrix.png + misclassified.json
.venv/bin/python visualize_predictions.py      # writes correct/incorrect_predictions.png; optional --num-examples --cols
.venv/bin/python -m pytest tests/              # protocol tests (shape, splits, val determinism, checkpoint) + behavior tests (loss math, seeding, checkpoint sanity)
.venv/bin/ruff check .                         # lint (ruff)
.venv/bin/ruff format .                        # format (ruff)
pyright .                                      # typecheck (pyright, config in pyproject.toml)
```
`train.py` argparse (defaults in `main()`): `--epochs 30`, `--batch-size 64`, `--lr 0.0008`,
`--seed 42`, `--patience 12`. `evaluate.py` takes optional `--out-dir`.
`visualize_predictions.py` takes optional `--out-dir`, `--num-examples` (default 8), `--cols` (default 4)
(writes the grids there; reads `misclassified.json` from the out dir).
Device selection (train/evaluate/visualize) comes from `utils.get_device()`: CUDA -> MPS -> CPU.

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
