import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _max_workers(configured: int) -> int:
    return min(configured, os.process_cpu_count())


def get_loaders(batch_size: int = 64, seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
    # Train loader uses light data augmentation for better generalization.
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(8),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    # Validation and test use the plain transform (no augmentation) so their
    # metrics are deterministic and comparable.
    plain_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(60_000, generator=generator).tolist()
    train_idx = perm[:55_000]
    val_idx = perm[55_000:60_000]

    train_dataset_aug = datasets.MNIST(
        root="data", train=True, download=True, transform=train_transform
    )
    train_dataset_plain = datasets.MNIST(
        root="data", train=True, download=True, transform=plain_transform
    )
    train_dataset = Subset(train_dataset_aug, train_idx)
    val_dataset = Subset(train_dataset_plain, val_idx)
    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=plain_transform
    )

    train_num_workers = int(os.environ.get("NUM_WORKERS", "16"))
    # pin_memory is only supported (and useful) on CUDA; enabling it on MPS/CPU
    # emits warnings.
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_max_workers(train_num_workers),
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_max_workers(4),
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_max_workers(4),
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
