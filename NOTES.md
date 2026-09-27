# NOTES.md — decision log (newest first)

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
