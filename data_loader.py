"""
data_loader.py – Centralized MNIST data loading.

All model files import get_loaders() from here.
MNIST is downloaded once to DATA_DIR (defined in config.py).
Subsequent calls simply load from disk with no network requests.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, MNIST_MEAN, MNIST_STD


def get_loaders(
    batch_size: int = BATCH_SIZE,
    augment: bool = True,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, test_loader) for MNIST.

    Args:
        batch_size:  Mini-batch size.
        augment:     If True, apply random affine + rotation to the training set.
                     Test set is never augmented.
        num_workers: Parallel workers for DataLoader.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    # ── Test transform (no augmentation) ──────────────────
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])

    # ── Train transform (optional augmentation) ───────────
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
    else:
        train_transform = base_transform

    # ── Download once; subsequent calls skip download ──────
    # datasets.MNIST checks for the raw files inside DATA_DIR.
    # If they exist, download=True is a no-op.
    train_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=base_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoader] Train: {len(train_dataset):,} | Test: {len(test_dataset):,} | "
          f"Batch: {batch_size} | Augment: {augment}")
    return train_loader, test_loader
