# AGENTS.md

## Project
PyTorch CNN for MNIST digit recognition.
Files: `data.py` (loaders/splits), `model.py` (MNISTNet), `train.py` (training loop),
`evaluate.py` (test eval, confusion matrix, misclassified.json), `visualize_predictions.py` (example grids).

## Environment
- Python 3.12 venv in `.venv/`. Run scripts with `.venv/bin/python`.
- GPU: RTX 3090. torch is the CUDA build (torch 2.14.0+cu130).
  **Never install CPU wheels (`+cpu`)** — this happened once and silently forced training onto the CPU.
  Training must print `Using device: cuda`.
- The GPU is shared with the Ollama model serving this session; only ~2-3 GB VRAM is free. Keep batch sizes modest (default 64).

## Commands
All from the project root:
```bash
.venv/bin/python train.py                      # optional: --epochs --batch-size --lr --seed --patience
.venv/bin/python evaluate.py                   # test eval, writes confusion_matrix.png + misclassified.json
.venv/bin/python visualize_predictions.py      # writes correct/incorrect_predictions.png
```
`train.py` argparse (defaults in `main()`): `--epochs 30`, `--batch-size 64`, `--lr 0.0008`,
`--seed 42`, `--patience 12`. `evaluate.py` and `visualize_predictions.py` take no arguments.

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
