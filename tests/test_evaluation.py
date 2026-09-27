"""Behavioral tests: evaluate_model math, seeding determinism, checkpoint sanity.

Distinct from test_protocol.py (structure/protocol): these guard the bugs this
project actually hit — per-batch loss averaging (fixed 2026-09-27) and a
best_model.pt that must actually be a trained model.
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import get_loaders
from evaluate import evaluate_model
from model import MNISTNet
from train import seed_everything

ROOT = Path(__file__).resolve().parent.parent


class _IndexedImageDataset(Dataset):
    """Image encodes its sample index (x[0,0,0] == i / 5); label pattern is known."""

    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.full((1, 28, 28), float(i) / 5.0), self.labels[i]


class _IndexAwareModel(nn.Module):
    """Out-of-distribution stand-in: log-prob of class 0 = -x[0,0,0], class 1 = -100,
    others -inf-equivalent (-1000). argmax is always 0; per-sample NLL varies with
    index, so unweighted per-batch averaging would give a different mean."""

    def forward(self, x):
        bs = x.size(0)
        out = torch.full((bs, 10), -1000.0)
        out[:, 0] = -x[:, 0, 0, 0]
        out[:, 1] = -100.0
        return out


def test_evaluate_model_weighted_loss_and_accuracy():
    # 5 samples in batches of 2, 2, 1 (last batch smaller — the old-bug trigger).
    labels = [0, 1, 0, 1, 0]
    loader = DataLoader(_IndexedImageDataset(labels), batch_size=2)
    model = _IndexAwareModel()
    loss, acc, preds, got_labels = evaluate_model(model, loader, torch.device("cpu"), nn.NLLLoss())
    assert acc == pytest.approx(3 / 5)
    assert got_labels == labels
    assert len(preds) == 5 and set(preds) == {0}
    # per-sample NLL: [0, 100, 0.4, 100, 0.8] -> weighted mean 201.2/5
    # old per-batch-average bug would yield (50 + 50.2 + 0.8)/3 = 33.667
    assert loss == pytest.approx(201.2 / 5.0, rel=1e-6)


def test_evaluate_model_loss_range_on_real_model():
    ckpt = ROOT / "best_model.pt"
    if not ckpt.exists():
        pytest.skip("no trained checkpoint yet")
    model = MNISTNet()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    _, val_loader, _ = get_loaders(batch_size=64)
    loss, acc, _, _ = evaluate_model(model, val_loader, torch.device("cpu"), nn.NLLLoss())
    assert loss is not None
    # Trained model on 5k validation images: loss must be finite, in [0, log(10)],
    # and substantially better than a random classifier (ln 10 ≈ 2.3026).
    assert math.isfinite(loss)
    assert 0.0 <= loss <= math.log(10)
    assert acc > 0.9


def test_seed_everything_reproducible_training():

    def run():
        seed_everything(123)
        net = nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        x = torch.randn(16, 1, 28, 28)
        y = torch.randint(0, 10, (16,))
        loss = F.nll_loss(net(x), y)
        for _ in range(3):
            opt.zero_grad()
            loss = F.nll_loss(net(x), y)
            loss.backward()
            opt.step()
        return {k: v.clone() for k, v in net.state_dict().items()}, float(loss.item())

    sd_a, loss_a = run()
    sd_b, loss_b = run()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"non-deterministic init/step: {key}"
    assert loss_a == loss_b
    # seed_everything covers random/np/torch — all three streams in sync
    from random import randint

    seed_everything(7)
    a1 = [randint(0, 100) for _ in range(5)]
    seed_everything(7)
    a2 = [randint(0, 100) for _ in range(5)]
    assert a1 == a2


def test_best_checkpoint_is_trained_not_random():
    ckpt = ROOT / "best_model.pt"
    if not ckpt.exists():
        pytest.skip("no trained checkpoint yet")
    model = MNISTNet()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    _, _, test_loader = get_loaders(batch_size=64)
    _, acc, _, _ = evaluate_model(model, test_loader, torch.device("cpu"))
    assert acc > 0.95, f"best_model.pt looks untrained (test acc {acc:.4f})"
