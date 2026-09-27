import argparse
import json
import math
import os
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from data import get_loaders
from utils import get_device, load_model

Example = tuple[Tensor, int, int]


def save_grid(
    items: Sequence[Example],
    filename: str,
    cols: int = 4,
    figsize_per_row: tuple[float, float] = (12, 6),
) -> None:
    if not items:
        print(f"No examples to save for {filename}")
        return
    rows = math.ceil(len(items) / cols)
    # constant tile size (old 8-item 4x2 grid was (12, 6))
    tile_w, tile_h = figsize_per_row[0] / 4.0, figsize_per_row[1] / 2.0
    fig, axes = plt.subplots(rows, cols, figsize=(tile_w * cols, tile_h * max(rows, 1)))
    axes = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
    for axis, (image, true_label, pred_label) in zip(axes, items, strict=False):
        axis.imshow(image.squeeze(0).cpu().numpy(), cmap="gray")
        axis.set_title(f"true: {true_label} | pred: {pred_label}")
        axis.axis("off")
    for axis in axes[len(items) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def collect_correct(
    model: torch.nn.Module, test_loader, device: torch.device, count: int
) -> list[Example]:
    examples: list[Example] = []
    for images, labels in test_loader:
        with torch.no_grad():
            log_probs = model(images.to(device))
            preds = log_probs.argmax(dim=1).cpu()
        for i in range(images.size(0)):
            if preds[i] == labels[i]:
                examples.append((images[i], int(labels[i]), int(preds[i])))
        if len(examples) >= count:
            break
    return examples[:count]


def collect_incorrect(
    preferred_indices: list[int] | None,
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    count: int,
) -> list[Example]:
    # Prefer the exact misclassified indices recorded by evaluate.py (reproducibility);
    # fall back to live inference if the JSON is missing or the model changed.
    if preferred_indices:
        indices = preferred_indices[:count]
        by_index: dict[int, tuple[Tensor, int]] = {}
        offset = 0
        for images, labels in test_loader:
            for i in range(images.size(0)):
                if offset + i in indices:
                    by_index[offset + i] = (images[i], int(labels[i]))
            offset += images.size(0)
            if len(by_index) >= len(indices):
                break
        if len(by_index) == len(indices):
            examples: list[Example] = []
            for idx in indices:
                image, true_label = by_index[idx]
                with torch.no_grad():
                    pred = int(model(image.to(device).unsqueeze(0)).argmax(dim=1).item())
                examples.append((image, true_label, pred))
            return examples

    examples = []
    for images, labels in test_loader:
        with torch.no_grad():
            log_probs = model(images.to(device))
            preds = log_probs.argmax(dim=1).cpu()
        for i in range(images.size(0)):
            if preds[i] != labels[i]:
                examples.append((images[i], int(labels[i]), int(preds[i])))
        if len(examples) >= count:
            break
    return examples[:count]


def preferred_misclassified_indices(mis_path: str, count: int) -> list[int] | None:
    if not os.path.exists(mis_path):
        return None
    with open(mis_path) as f:
        misclassified = json.load(f)
    if not misclassified:
        return None
    return [int(entry["index"]) for entry in misclassified[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize sample predictions")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="directory for correct_predictions.png and incorrect_predictions.png",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="number of correct/incorrect examples to show per grid (default 8)",
    )
    parser.add_argument("--cols", type=int, default=4, help="columns per grid (default 4)")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    model = load_model("best_model.pt", device)
    model.eval()

    # Deterministic, un-augmented test data (test loader in get_loaders)
    _, _, test_loader = get_loaders(batch_size=64)

    correct_examples = collect_correct(model, test_loader, device, args.num_examples)
    mis_path = os.path.join(out_dir, "misclassified.json")
    indices = preferred_misclassified_indices(mis_path, args.num_examples)
    incorrect_examples = collect_incorrect(indices, model, test_loader, device, args.num_examples)

    print(
        f"Collected {len(correct_examples)} correct, {len(incorrect_examples)} incorrect examples"
    )

    save_grid(correct_examples, os.path.join(out_dir, "correct_predictions.png"), cols=args.cols)
    save_grid(
        incorrect_examples, os.path.join(out_dir, "incorrect_predictions.png"), cols=args.cols
    )


if __name__ == "__main__":
    main()
