from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import PlantDiseaseDataset
from src.data.transforms import train_transform, val_transform
from src.models.cnn_experiment_02 import PlantDiseaseCNNExperiment02
from src.training.losses import create_class_weights
from src.training.trainer import Trainer


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_manifest.csv"

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "experiments"
    / "experiment_02"
)

LOG_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiments"
    / "experiment_02_history.csv"
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

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SMART AGRI - EXPERIMENT 02")
print("Batch Normalization")
print("=" * 60)

print(f"Device: {device}")


# ============================================================
# Dataset
# ============================================================

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


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


print(f"Training samples   : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")
print(f"Batch size         : {BATCH_SIZE}")


# ============================================================
# Model
# ============================================================

model = PlantDiseaseCNNExperiment02(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# ============================================================
# Class weights
# ============================================================

class_weights = create_class_weights(
    manifest_path=MANIFEST_PATH,
    num_classes=NUM_CLASSES,
)

class_weights = class_weights.to(device)


# ============================================================
# Loss function
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
# Learning-rate scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
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
    log_file=LOG_PATH,
    gradient_clip=GRADIENT_CLIP,
)


# ============================================================
# Training
# ============================================================

trainer.fit(
    epochs=EPOCHS,
    patience=PATIENCE,
)


print()
print("=" * 60)
print("EXPERIMENT 02 COMPLETE")
print("=" * 60)

print(f"Best checkpoint: {CHECKPOINT_DIR / 'best_model.pt'}")
print(f"Training history: {LOG_PATH}")