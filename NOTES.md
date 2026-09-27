# NOTES.md — decision log (newest first)

## 2026-09-27 — Generated outputs moved to a `results/` subfolder
- All script outputs (training curves + CSV, confusion matrix + misclassified.json,
  prediction grids, feature figures) now default to `results/`; `--out-dir` still
  overrides per run (e.g. `results/run1`). `train.py` gained a small `_ensure_parent_dir`
  helper (matplotlib/CSV do not create parent dirs). Existing committed PNGs +
  misclassified.json + training_history.csv moved into `results/` via `git mv`.

## 2026-09-27 — visualize_features.py added (conv1 kernels + per-digit activation maps)
New script `visualize_features.py` (make target: `make features`) for inspecting learned
features, loading `best_model.pt` read-only (eval mode, `torch.no_grad`):
- `conv1_filters.png`: conv1's 32 learned 3x3 kernels as a 4x8 grid (per-kernel
  min/max normalization to a shared [0,1], grayscale).
- `feature_maps.png`: one test image per digit 0-9 (from the un-augmented test set),
  with 8 evenly-spaced channels sampled from the output of each of the three conv
  blocks (grayscale), plus the model's predicted digit per row.
Implementation notes: block outputs are captured with a single forward hook on the
**shared** `model.pool` module (called once per block), which fires exactly 3x in block
order during one forward, giving (1,32,14,14) / (1,64,7,7) / (1,128,3,3) activations.
Display inputs are denormalized (undo `Normalize(0.1307, 0.3081)`) to [0,1].
No changes to model architecture, hyperparameters, augmentation, or the test-set
protocol (test images are only read for display, not used for any tuning).
Verified: runs on CUDA, both PNGs produced; `ruff check` clean, `pyright` 0 errors,
`pytest` 8/8.

## 2026-09-27 — Engineering merge from the `mnist` sibling repo
Folded the engineering strengths of the earlier `jtompuri/mnist` repo into this
repo, keeping this repo's (better) ML setup untouched:
- **MPS support** (Apple Silicon): new `utils.get_device()` CUDA -> MPS -> CPU, now
  used by train/evaluate/visualize. `utils.set_seed()` also seeds MPS. This repo
  previously hard-coded `cuda or cpu` and would have silently trained on CPU on a
  Mac.
- **Shared `utils.py`**: `get_device`, `set_seed`, `load_model(weights_only=True)`.
  `train.seed_everything` is now an alias for `utils.set_seed` (tests still import it).
- **Type hints** added across model/data/train/evaluate/visualize/utils.
- **Packaging**: `pyproject.toml` is now the single source of truth (added
  `[build-system]`, `dev` extra = pytest+ruff+pyright, `[tool.ruff]`); `requirements.txt`
  removed; install is `pip install -e ".[dev]"`.
- **Makefile**: `setup train eval viz test lint format typecheck check`.
- **Code quality**: ruff (lint+format) and pyright wired in; CI switched from
  `requirements.txt + pytest` to `pip install -e ".[dev]"` + `ruff check` + `pyright` + pytest.
- **visualize_predictions.py**: `--num-examples` (default 8) and adaptive grid rows.
Verified locally: `ruff check` clean, `pyright` 0 errors, `pytest` 8/8 pass.
No architecture/hyperparameter/augmentation changes.

## 2026-09-27 — visualize_predictions.py gained --out-dir
Symmetry with `evaluate.py`: grids can now be written to a custom directory, and the
`misclassified.json` lookup uses that same directory (fallback: cwd). Default behavior
unchanged (`.`). Verified: `--out-dir /tmp/...` writes both PNGs there. Remaining
deliberate non-features: no `--batch-size` on evaluate (one-shot, accuracy unaffected),
no per-class reporting / TorchScript export / predict CLI (out of scope for this project).

## 2026-09-27 — CI added; train.py inline shape check removed
Added `.github/workflows/ci.yml` (ubuntu-latest, py3.14, `pytest tests/`) so the
pre-commit test rule is enforced automatically. Removed the inline shape/log_softmax
check in `train.py` (duplicated `test_protocol.py::test_model_shape_and_log_softmax`).
Note on `best_model.pt`: it is NOT bit-reproducible from source — augmentation RNG
depends on worker/thread interleaving per run (see nw=16 entry), so reruns land at
val 0.0123–0.0145 / test 99.63–99.64%. The committed artifact is the 99.64% one.

## 2026-09-27 — Behavioral tests added (loss math, seeding, checkpoint guard)
Question "are there enough tests?" prompted a review: `test_protocol.py` covered structure
only, while the bugs this project actually hit were behavioral. Added
`tests/test_evaluation.py`: (1) `evaluate_model` sample-weighted loss + accuracy against a
synthetic index-aware model with unequal batch sizes (2, 2, 1) — a uniform-loss test would
NOT catch the old per-batch-averaging bug, which this one does (old code: 33.67 vs correct
40.24); (2) `seed_everything` reproduces init + 3 optimizer steps bit-for-bit;
(3) val-loss range [0, ln 10] on the real checkpoint; (4) `best_model.pt` must score > 95%
test accuracy so a corrupted/random checkpoint file cannot masquerade as trained.
Suite now 8 tests, all passing.

## 2026-09-27 — Train-loader workers 8 → 16
Made train-loader worker count tunable via `NUM_WORKERS` env var (default 16) in `data.py`.
A/B (seed 42, 30 epochs, full runs):
- nw=8:  wall 111.15s,  best val_loss 0.0123 @ ep29, test **99.64%** (36)
- nw=16: wall **87.91s**, best val_loss 0.0137 @ ep27, test **99.63%** (37)
Run-to-run variance: same-config nw=16 reruns give val_loss 0.0136→0.0145→0.0141 and
test ±1 sample — the CPU-RNG interleaving that seeds RandomRotation/RandomAffine differs
each run, so val_loss wanders ~0.0123–0.0145 and test sits at 99.63–99.64%.
Decision: keep **nw=16** (~22% faster, no meaningful loss on the final test metric).
Seq. re-test of higher worker counts (16-core/32-thread Ryzen 9 5950X), run one at a time:
nw=16 → 88.4s, nw=24 → 88.7s, nw=32 → 91.3s → **no gain above 16**; SMT threads don't
help a GPU-consumption-bound pipeline, extra workers only add scheduler overhead.
Final committed model (nw=16 rerun): best val_loss 0.0141 @ ep27, test **99.64%** (36).
Note: box is shared with Ollama; 16 is fine with current headroom but could contend under load.

## 2026-09-27 — GPU augmentation investigated, NOT adopted (reverted)
Implemented `augment_train()` in `data.py` (RandomRotation(8) + RandomAffine translate via
`grid_sample` on CUDA, mirroring the CPU pipeline). A/B (same seed, same hyperparams, 30
epochs): CPU aug best val_loss **0.0125** @ ep29, test **99.62%** (~155s total); GPU aug
best val_loss **0.0242** @ ep29, test **99.30%** (~84s total). GPU side was ~2x faster per
epoch but ~2px worse on the final test metric — an unexplained quality/speed trade-off, not
just variance. Reverted to the CPU pipeline (quality kept, speed already covered by the
worker-tuning entry below). If revisiting: ensure the warp distribution statistically
matches torchvision's (p=0.5 skip probability) before re-adopting.

## 2026-09-27 — Train loader made data-loading-bound-friendly (nw=8, persistent, prefetch=4)
Diagnosis: a bare pass over `train_loader` (no model, just moving 860 batches to GPU) ran
6.22s vs a full epoch 6.29s → the data pass is ~99% of an epoch, i.e. training is
**data-loading bound**, not compute bound (the old ~2.7%-data figure was a per-step
measurement artifact of Ollama GPU contention + per-step syncs). Applied to the TRAIN
loader only: `num_workers=4→8`, `persistent_workers=True`, `prefetch_factor=4`.
Result: full epoch 6.29s → **~3.3s** (~2x faster). No batch-size/augmentation/hyperparameter
changes. Still data-bound at ~99% of the epoch; the next lever (not applied) is moving
augmentation to the GPU.

## 2026-09-27 — Test suite + eval `--out-dir` added
Added `tests/test_protocol.py` (pytest): model shape/log_softmax, 55k/5k/10k split sizes,
validation loader determinism (no augmentation), and `best_model.pt` round-trip into a
fresh `MNISTNet` (guards against the `_orig_mod.` prefix gotcha found during the
torch.compile investigation). `evaluate.py` gained `--out-dir` (default `.`).
`pyproject.toml` now declares project dependencies + a `dev` extra (pytest).

## 2026-09-27 — torch.compile investigated, NOT adopted
Profile showed low GPU utilization (~17%) and ~97% of epoch time in per-step compute
wall (data loading only 2.7%), suggesting Python/kernel-launch overhead could be
reduced with `torch.compile`. A/B benchmark (warm, same process, 60k images, batch 64):
eager **6.94s/epoch** vs compiled **7.11s/epoch** (first epoch 7.38s incl. one-time
compile) — no speedup, slight regression, consistent with the small model size and
the GPU being shared with Ollama. Also verified: saving `compiled_model.state_dict()`
writes `_orig_mod.`-prefixed keys that a plain `MNISTNet.load_state_dict()` rejects,
so integration would have required `_orig_mod`-safe checkpointing in train/eval/visualize.
Decision (user-approved): do not integrate `torch.compile`; no code changes.
Note: the "97% compute / 2.7% data" profile here was later shown to be a
measurement artifact (Ollama contention + per-step syncs) — the nw=8 entry above
found training was actually data-loading bound; the compile decision stands on the
A/B speedup result (~no gain), independent of that profiling.

## 2026-09-27 — patience raised to 12 so cosine LR schedule runs its course (resolves open issue)
Changed `train.py` default `--patience` from 5 to 12 (best-checkpoint selection still
on val_loss, unchanged). Training now completes all 30 epochs instead of early-stopping
at epoch 10, so `CosineAnnealingLR(T_max=30)` fully decays from 0.0008 to ~0.
Result (`training_history.csv`): best val_loss 0.0134 at epoch **29**
(best val_acc **99.52%**), final test accuracy **99.62%** (up from 99.36% at patience 5).

## 2026-09-27 — Train/val/test protocol replaces test-set checkpoint selection
Selecting the "best" checkpoint from test-set accuracy leaked test information
into model selection, so the earlier 99.60% test result (60k train, no val split)
was optimistic. Switched to 55k train / 5k val (seed 42), no augmentation on val,
val-only checkpointing + early stopping. Test set now evaluated once, in `evaluate.py`.
Result (`training_history.csv`): best val_loss 0.02376 at epoch 5 (val_acc 99.24%);
early stopped at epoch 10 (patience 5); final test accuracy **99.36%**.

## 2026-09-27 — ~~Open issue: early stopping vs cosine LR schedule~~ (RESOLVED)
Early stopping at `--patience 5` triggered on validation noise (val_loss spiked back up
from 0.0238 to ~0.038 at epoch 6) and cut `CosineAnnealingLR(T_max=30)` short at
epoch 10, before the schedule meaningfully decays from the 0.0008 start.
Resolved 2026-09-27: default patience raised to 12 (see entry above); training now
completes the full 30-epoch cosine schedule.

## 2026-09-27 — torch switched from CPU build to CUDA build
The CPU wheel had been installed at some point, silently forcing training onto the CPU.
Reinstalled the CUDA build (torch 2.14.0+cu130, RTX 3090). Rule added to AGENTS.md:
never install `+cpu` wheels; training must print `Using device: cuda`.

## 2026-09-27 — 3rd conv block added (approved deviation)
Model grew from 2 to 3 conv blocks: 1→32→64→**128**, each `Conv2d(3,3,pad=1)` + BN + ReLU + MaxPool,
then dropout 0.25, fc 128·3·3→128, fc 128→10, log_softmax. This deviates from the original 2-conv spec — approved by the user.
Augmentation in `data.py` is currently `RandomRotation(8)` + `RandomAffine(degrees=0, translate=(0.08, 0.08))`;
these were **changed from the originally approved parameters** (rotation 15, translate 0.12, scale 0.95–1.05).
Current values are treated as the de-facto standard; restore the approved values if reverting.

## 2026-09-27 — evaluate_model() averaging bug fixed
The shared evaluator previously averaged per-batch losses unweighted, so epoch loss
was wrong when the last batch was smaller than the others. Fixed to accumulate
`loss * batch_size` and divide by total samples (same in `train.py` via `seen` accumulator).

## 2026-09-27 — Code quality improvements batch
Added over the baseline: data augmentation, BatchNorm, early stopping, CosineAnnealingLR,
deterministic seeding (`seed_everything`, seed 42), training curves (PNG + `training_history.csv`),
shared `evaluate_model()` helper, argparse in `train.py`, pinned `requirements.txt`,
row-normalized confusion matrix panel, and `misclassified.json` for reproducible inspection.

## 2026-09-27 — Baseline
Original 2-conv model reached **99.08%** test accuracy; used as the reference before the
improvement batch above.
