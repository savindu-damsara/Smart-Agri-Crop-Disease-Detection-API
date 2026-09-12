from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import PlantDiseaseDataset

from src.data.transforms import (
    train_transform,
    val_transform,
)

from src.models.cnn import (
    PlantDiseaseCNN
)

from src.training.trainer import (
    Trainer
)

from src.training.losses import (
    create_class_weights
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_manifest.csv"
)


CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
)


LOG_FILE = (
    PROJECT_ROOT
    / "reports"
    / "experiments"
    / "training_history.csv"
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32

NUM_CLASSES = 38

EPOCHS = 20

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

PATIENCE = 5

NUM_WORKERS = 0

SEED = 42

GRADIENT_CLIP = 1.0


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


set_seed(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 60)

print(
    "SMART AGRI - PYTORCH TRAINING"
)

print("=" * 60)

print()

print(
    f"Device: {device}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print()


# ============================================================
# Dataset
# ============================================================

print(
    "Loading datasets..."
)


train_dataset = PlantDiseaseDataset(
    manifest_path=MANIFEST_PATH,
    split="train",
    transform=train_transform,
)


val_dataset = PlantDiseaseDataset(
    manifest_path=MANIFEST_PATH,
    split="validation",
    transform=val_transform,
)


print(
    f"Training samples: "
    f"{len(train_dataset)}"
)


print(
    f"Validation samples: "
    f"{len(val_dataset)}"
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(
        device.type == "cuda"
    ),
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        device.type == "cuda"
    ),
)


# ============================================================
# Model
# ============================================================

print(
    "\nCreating model..."
)


model = PlantDiseaseCNN(
    num_classes=NUM_CLASSES
)


model = model.to(device)


print(
    "Model created successfully."
)


# ============================================================
# Class Weights
# ============================================================

print(
    "\nCalculating class weights..."
)


class_weights = create_class_weights(
    MANIFEST_PATH,
    NUM_CLASSES,
)


class_weights = class_weights.to(
    device
)


print(
    "Class weights calculated."
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Learning Rate Scheduler
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    checkpoint_dir=CHECKPOINT_DIR,
    log_file=LOG_FILE,
    gradient_clip=GRADIENT_CLIP,
)


# ============================================================
# Start Training
# ============================================================

trainer.fit(
    epochs=EPOCHS,
    patience=PATIENCE,
)