"""Shared device selection, seeding, and model-loading helpers.

Used by train.py, evaluate.py and visualize_predictions.py so device and
seeding logic is defined once. Device selection falls through CUDA -> MPS
(Apple Silicon) -> CPU.
"""

import random

import numpy as np
import torch

from model import MNISTNet

DEFAULT_SEED = 42


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_model(path="best_model.pt", device: torch.device | None = None) -> MNISTNet:
    device = device or get_device()
    model = MNISTNet()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)
