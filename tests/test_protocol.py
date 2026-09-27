from pathlib import Path

import torch

from data import get_loaders
from model import MNISTNet

ROOT = Path(__file__).resolve().parent.parent


def test_model_shape_and_log_softmax():
    model = MNISTNet()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)
    # log_softmax output: exponentiated rows sum to 1
    assert torch.allclose(out.exp().sum(dim=1), torch.ones(4), atol=1e-5)


def test_loader_split_sizes():
    train_loader, val_loader, test_loader = get_loaders(batch_size=64)
    n_train = sum(img.shape[0] for img, _ in train_loader)
    n_val = sum(img.shape[0] for img, _ in val_loader)
    n_test = sum(img.shape[0] for img, _ in test_loader)
    assert (n_train, n_val, n_test) == (55_000, 5_000, 10_000)


def test_val_loader_is_deterministic_no_augmentation():
    # Protocol: validation uses the plain (un-augmented) transform, so two
    # consecutive loads of the same batch must be identical.
    _, val_loader, _ = get_loaders(batch_size=64)
    first = next(iter(val_loader))
    second = next(iter(val_loader))
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_best_checkpoint_loads_into_model():
    ckpt = ROOT / "best_model.pt"
    if not ckpt.exists():
        return  # nothing trained yet; CI environments without artifacts
    model = MNISTNet()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()
