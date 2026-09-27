import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from data import get_loaders
from utils import get_device, load_model


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float | None, float, list[int], list[int]]:
    """Shared evaluation: returns (avg_loss, acc, all_preds, all_labels).

    avg_loss is None if criterion is not given.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            log_probs = model(images)
            preds = log_probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            correct += int((preds == labels.numpy()).sum())
            if criterion is not None:
                total_loss += criterion(log_probs, labels.to(device)).item() * images.size(0)
            total += images.size(0)
    avg_loss = total_loss / total if criterion is not None else None
    return avg_loss, correct / total, all_preds, all_labels


def save_confusion_matrices(
    all_labels: list[int],
    all_preds: list[int],
    accuracy: float,
    out_path: str = "confusion_matrix.png",
) -> np.ndarray:
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    _annotate_cm(axes[0], cm, thresholds=cm.max() / 2.0)

    im = axes[1].imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=axes[1])
    axes[1].set_title("Confusion Matrix (row-normalized)")
    _annotate_cm(axes[1], cm_norm, thresholds=0.5, fmt=".1%")

    for ax in axes:
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))

    fig.suptitle(f"Overall accuracy: {100 * accuracy:.2f}%")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def _annotate_cm(ax: Axes, cm: np.ndarray, thresholds: float, fmt: str = "d") -> None:
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresholds else "black",
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate best model on the test set")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="directory for confusion_matrix.png and misclassified.json",
    )
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    model = load_model("best_model.pt", device)

    _, _, test_loader = get_loaders(batch_size=64)
    _, accuracy, all_preds, all_labels = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {100 * accuracy:.2f}%")

    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    cm = save_confusion_matrices(all_labels, all_preds, accuracy, cm_path)
    print(f"Confusion matrix:\n{cm}")
    print(f"Saved {cm_path}")

    misclassified = [
        {"index": i, "true_label": t, "predicted_label": p}
        for i, (t, p) in enumerate(zip(all_labels, all_preds))
        if t != p
    ]
    mis_path = os.path.join(out_dir, "misclassified.json")
    with open(mis_path, "w") as f:
        json.dump(misclassified, f, indent=2)
    print(f"Saved {mis_path} ({len(misclassified)} samples)")


if __name__ == "__main__":
    main()
