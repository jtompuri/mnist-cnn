"""Conv-kernel and convolution-block visualization for MNISTNet.

Loads the best model (``best_model.pt``) and writes two figures to
``--out-dir`` (default, project root):

- ``conv1_filters.png``: ``conv1``'s 32 learned 3x3 kernels as a viridis grid.
- ``feature_maps.png``: for one test image per digit, the activation maps after
  each of the three conv blocks (8 evenly-spaced channels sampled per block)
  plus the digit the model predicted.

No training or weight modifications are performed.
"""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes

from data import get_loaders
from model import MNISTNet
from utils import get_device, load_model

DEFAULT_MODEL_PATH = "best_model.pt"
CHANNELS_PER_BLOCK = 8
BLOCK_LABELS = ("block 1 - 32 ch", "block 2 - 64 ch", "block 3 - 128 ch")


def sample_channels(num_channels: int, num_to_show: int) -> list[int]:
    """Evenly spaced channel indices spanning first..last, deduplicated."""
    if num_to_show >= num_channels:
        return list(range(num_channels))
    indices = [round(i * (num_channels - 1) / (num_to_show - 1)) for i in range(num_to_show)]
    seen: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.append(idx)
    return seen


def denormalize(image: torch.Tensor, mean: float = 0.1307, std: float = 0.3081) -> torch.Tensor:
    """Undo Normalize so pixel values fall in [0, 1] for display."""
    tensor = image.clone()
    tensor *= std
    tensor += mean
    return tensor.clamp(0.0, 1.0)


def predict_digit(model: MNISTNet, image: torch.Tensor, device: torch.device) -> int:
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
    return int(logits.argmax(dim=1).item())


def show_tile(ax: Axes, tensor: torch.Tensor, cmap: str) -> None:
    ax.imshow(tensor.squeeze().detach().cpu().numpy(), cmap=cmap)
    ax.axis("off")
    ax.set_aspect("equal")


def plot_conv1_filters(model: MNISTNet, out_dir: Path) -> None:
    weights = model.conv1.weight.detach().cpu().numpy()  # (32, 1, 3, 3)
    filters = weights[:, 0]  # (32, 3, 3)
    rows, cols = 4, 8
    lower = float(filters.min())
    upper = float(filters.max())
    span = max(upper - lower, 1e-8)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.8, rows * 0.8))
    for index, ax in enumerate(axes.flat):
        data = filters[index].astype("float64")
        normalized = (data - lower) / span
        ax.imshow(normalized, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"{index}", fontsize=8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.suptitle(f"conv1 kernels ({filters.shape[0]}x3x3, shared vmin/vmax)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "conv1_filters.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'conv1_filters.png'}")


def capture_blocks(
    model: MNISTNet, image: torch.Tensor, device: torch.device
) -> list[torch.Tensor]:
    """Capture the output of each conv block (the shared pool) in block order."""
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: object, output: torch.Tensor) -> None:
        captured.append(output.detach())

    # `model.pool` is shared and called once per block; hooking it fires 3x in
    # forward order during a single forward pass.
    handle = model.pool.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(image.unsqueeze(0).to(device))
    finally:
        handle.remove()
    return captured[:3]


def grab_one_image_per_digit(test_loader, device: torch.device) -> list[torch.Tensor]:
    images: dict[int, torch.Tensor] = {}
    for batch_images, labels in test_loader:
        for i in range(batch_images.size(0)):
            digit = int(labels[i].item())
            if digit not in images:
                images[digit] = batch_images[i].to(device).clone()
            if len(images) == 10:
                return [images[d] for d in range(10)]
    raise RuntimeError("Test set did not contain all ten digits")


def plot_feature_maps(
    model: MNISTNet, images: Sequence[torch.Tensor], device: torch.device, out_dir: Path
) -> None:
    rows, cols_per_block = 10, CHANNELS_PER_BLOCK
    col_starts = [1 + i * cols_per_block for i in range(3)]
    total_cols = 1 + len(col_starts) * cols_per_block

    fig, axes = plt.subplots(rows, total_cols, figsize=(total_cols * 0.55, rows * 0.55))
    for digit in range(rows):
        image = images[digit]
        predicted = predict_digit(model, image, device)
        blocks = [b.squeeze(0) for b in capture_blocks(model, image, device)]

        show_tile(axes[digit, 0], denormalize(image), "gray")
        axes[digit, 0].set_ylabel(f"true {digit} | pred {predicted}", fontsize=8)

        for block_idx, block in enumerate(blocks):
            num_channels = block.shape[0]
            sample = sample_channels(num_channels, cols_per_block)
            for offset, channel in enumerate(sample):
                show_tile(axes[digit, col_starts[block_idx] + offset], block[channel], "magma")

    for block_idx, label in enumerate(BLOCK_LABELS):
        axes[0, col_starts[block_idx]].set_title(label, fontsize=8)
    axes[0, 0].set_title("input", fontsize=8)

    fig.suptitle("MNISTNet activation maps per digit (8 sampled channels per block)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "feature_maps.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'feature_maps.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=str, default="results", help="Directory for output figures (default 'results')"
    )
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to best_model.pt"
    )
    args = parser.parse_args()

    out_dir = Path(os.path.abspath(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    model = load_model(args.model_path, device)
    model.eval()

    _, _, test_loader = get_loaders(batch_size=64)
    images = grab_one_image_per_digit(test_loader, device)

    plot_conv1_filters(model, out_dir)
    plot_feature_maps(model, images, device, out_dir)


if __name__ == "__main__":
    main()
