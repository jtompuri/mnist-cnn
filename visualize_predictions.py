import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from data import get_loaders
from model import MNISTNet


def save_grid(items, filename):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, (image, true_label, pred_label) in zip(axes.flatten(), items):
        ax.imshow(image.squeeze(0).cpu().numpy(), cmap="gray")
        ax.set_title(f"true: {true_label} | pred: {pred_label}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize sample predictions")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="directory for correct_predictions.png and incorrect_predictions.png")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MNISTNet().to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    # Deterministic, un-augmented test data (test loader in get_loaders)
    _, _, test_loader = get_loaders(batch_size=64)

    correct_examples = []
    for images, labels in test_loader:
        with torch.no_grad():
            log_probs = model(images.to(device))
            preds = log_probs.argmax(dim=1).cpu()
        for i in range(images.size(0)):
            if preds[i] == labels[i]:
                correct_examples.append((images[i], int(labels[i]), int(preds[i])))
        if len(correct_examples) >= 8:
            break
    correct_examples = correct_examples[:8]

    # Prefer the exact misclassified indices recorded by evaluate.py (reproducibility);
    # fall back to live inference if the JSON is missing or the model changed.
    mis_path = os.path.join(out_dir, "misclassified.json")
    incorrect_examples = None
    if os.path.exists(mis_path):
        with open(mis_path) as f:
            misclassified = json.load(f)
        if misclassified:
            indices = [entry["index"] for entry in misclassified[:8]]
            by_index = {}
            offset = 0
            for images, labels in test_loader:
                for i in range(images.size(0)):
                    if offset + i in indices:
                        by_index[offset + i] = (images[i], int(labels[i]))
                offset += images.size(0)
                if len(by_index) >= len(indices):
                    break
            if len(by_index) == len(indices):
                incorrect_examples = []
                for idx in indices:
                    image, true_label = by_index[idx]
                    with torch.no_grad():
                        pred = int(model(image.to(device).unsqueeze(0)).argmax(dim=1).item())
                    incorrect_examples.append((image, true_label, pred))

    if incorrect_examples is None:
        incorrect_examples = []
        for images, labels in test_loader:
            with torch.no_grad():
                log_probs = model(images.to(device))
                preds = log_probs.argmax(dim=1).cpu()
            for i in range(images.size(0)):
                if preds[i] != labels[i]:
                    incorrect_examples.append((images[i], int(labels[i]), int(preds[i])))
            if len(incorrect_examples) >= 8:
                break
        incorrect_examples = incorrect_examples[:8]

    print(f"Collected {len(correct_examples)} correct, "
          f"{len(incorrect_examples)} incorrect examples")

    save_grid(correct_examples, os.path.join(out_dir, "correct_predictions.png"))
    save_grid(incorrect_examples, os.path.join(out_dir, "incorrect_predictions.png"))


if __name__ == "__main__":
    main()
