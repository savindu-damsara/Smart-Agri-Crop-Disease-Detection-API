from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "data")
)

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "models")
)

from src.data.dataset import PlantDiseaseDataset

from src.data.transforms import (
    train_transform
)

from src.models.cnn import PlantDiseaseCNN


 
# Configuration


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_manifest.csv"
)

BATCH_SIZE = 32

NUM_CLASSES = 38



# Device


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 60)
print("SMART AGRI - MODEL + REAL DATA TEST")
print("=" * 60)

print(
    f"\nDevice: {device}"
)



# Dataset


train_dataset = PlantDiseaseDataset(
    manifest_path=MANIFEST_PATH,
    split="train",
    transform=train_transform,
)



# DataLoader


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)



# Model


model = PlantDiseaseCNN(
    num_classes=NUM_CLASSES
)


model = model.to(device)



# Get One Batch


images, labels = next(
    iter(train_loader)
)


images = images.to(device)

labels = labels.to(device)


print(
    f"\nImages shape: {images.shape}"
)

print(
    f"Labels shape: {labels.shape}"
)


# Forward Pass


outputs = model(
    images
)


print(
    f"Outputs shape: {outputs.shape}"
)


# Prediction

predictions = torch.argmax(
    outputs,
    dim=1
)


print(
    f"\nPredictions: {predictions[:10].tolist()}"
)

print(
    f"Actual labels: {labels[:10].tolist()}"
)



# Success

print(
    "\nReal dataset successfully passed through CNN!"
)