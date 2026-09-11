from pathlib import Path

from torch.utils.data import DataLoader

from dataset import PlantDiseaseDataset
from transforms import (
    train_transform,
    validation_transform,
)


# Configuration


MANIFEST_PATH = Path(
    "data/processed/dataset_manifest.csv"
)

BATCH_SIZE = 32


# Create Datasets


train_dataset = PlantDiseaseDataset(
    manifest_path=MANIFEST_PATH,
    split="train",
    transform=train_transform,
)


validation_dataset = PlantDiseaseDataset(
    manifest_path=MANIFEST_PATH,
    split="validation",
    transform=validation_transform,
)


test_dataset = PlantDiseaseDataset(
    manifest_path=MANIFEST_PATH,
    split="test",
    transform=validation_transform,
)


# Create DataLoaders


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# Display Information


print("=" * 60)
print("PYTORCH DATALOADER TEST")
print("=" * 60)

print(
    f"Training samples   : {len(train_dataset):,}"
)

print(
    f"Validation samples : {len(validation_dataset):,}"
)

print(
    f"Test samples       : {len(test_dataset):,}"
)

print(
    f"Batch size         : {BATCH_SIZE}"
)

# Get One Batch


images, labels = next(
    iter(train_loader)
)


print(
    f"\nImage batch shape  : {images.shape}"
)

print(
    f"Label batch shape  : {labels.shape}"
)

print(
    f"First labels       : {labels[:10].tolist()}"
)

print("\nDataLoader test successful!")