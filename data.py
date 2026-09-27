import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_loaders(batch_size=64, seed=42):
    # Train loader uses light data augmentation for better generalization.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(8),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    # Validation and test use the plain transform (no augmentation) so their
    # metrics are deterministic and comparable.
    plain_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(60_000, generator=generator).tolist()
    train_idx = perm[:55_000]
    val_idx = perm[55_000:60_000]

    train_dataset_aug = datasets.MNIST(root="data", train=True, download=True, transform=train_transform)
    train_dataset_plain = datasets.MNIST(root="data", train=True, download=True, transform=plain_transform)
    train_dataset = Subset(train_dataset_aug, train_idx)
    val_dataset = Subset(train_dataset_plain, val_idx)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=plain_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return train_loader, val_loader, test_loader
