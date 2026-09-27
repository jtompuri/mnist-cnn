import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from data import get_loaders
from evaluate import evaluate_model
from model import MNISTNet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_curves(history, out_path="training_curves.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [100 * h["val_acc"] for h in history], marker="o")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)

    best = min(history, key=lambda h: h["val_loss"])
    axes[1].axvline(best["epoch"], color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_history_csv(history, out_path="training_history.csv"):
    fieldnames = ["epoch", "train_loss", "val_loss", "val_acc", "lr"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MNISTNet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5,
                        help="early stopping patience (epochs without best val_loss)")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MNISTNet().to(device)

    # Quick shape verification before training: input (4,1,28,28) -> output (4,10)
    with torch.no_grad():
        out = model(torch.randn(4, 1, 28, 28, device=device))
    assert out.shape == (4, 10), f"Bad output shape: {out.shape}"
    assert torch.allclose(out.exp().sum(dim=1), torch.ones(4, device=out.device), atol=1e-5), \
        "log_softmax not valid"
    print(f"Shape check passed: input (4,1,28,28) -> output {tuple(out.shape)}")

    train_loader, val_loader, _ = get_loaders(batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.NLLLoss()  # model outputs log_softmax -> NLLLoss, NOT CrossEntropyLoss

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        train_loss = running_loss / seen

        val_loss, val_acc, _, _ = evaluate_model(
            model, val_loader, device, criterion
        )
        assert val_loss is not None  # criterion was passed, so a loss is returned
        print(f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {100 * val_acc:.2f}% | "
              f"lr: {optimizer.param_groups[0]['lr']:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  -> saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  -> early stopping: no improvement for {args.patience} epochs")
                break

        scheduler.step()

    save_history_csv(history)
    plot_curves(history)

    if os.path.exists("best_model.pt"):
        print(f"Done. Best val_loss {best_val_loss:.4f} at epoch {best_epoch}. "
              f"Saved to best_model.pt")
    else:
        raise RuntimeError("No model was saved (no epoch completed).")


if __name__ == "__main__":
    main()
