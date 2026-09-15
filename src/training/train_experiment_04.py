from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import PlantDiseaseDataset
from src.models.resnet18_transfer import PlantDiseaseResNet18
from src.training.losses import create_class_weights
from src.training.trainer import Trainer


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    / "experiments"
    / "experiment_04"
)

LOG_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiments"
    / "experiment_04_history.csv"
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 32
NUM_CLASSES = 38
EPOCHS = 20

# Lower learning rate for fine-tuning
LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001
PATIENCE = 5

NUM_WORKERS = 0
SEED = 42
GRADIENT_CLIP = 1.0


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# --------------------------------------------------
# TRANSFORMS
# --------------------------------------------------

IMAGE_SIZE = 224

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# --------------------------------------------------
# DATASETS
# --------------------------------------------------

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


# --------------------------------------------------
# DATALOADERS
# --------------------------------------------------

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


print(f"Training samples   : {len(train_dataset):,}")
print(f"Validation samples : {len(val_dataset):,}")


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = PlantDiseaseResNet18(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# --------------------------------------------------
# CLASS WEIGHTS
# --------------------------------------------------

class_weights = create_class_weights(
    manifest_path=MANIFEST_PATH,
    num_classes=NUM_CLASSES,
)

class_weights = class_weights.to(device)


# --------------------------------------------------
# LOSS
# --------------------------------------------------

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# --------------------------------------------------
# OPTIMIZER
# --------------------------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# --------------------------------------------------
# LR SCHEDULER
# --------------------------------------------------

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# --------------------------------------------------
# TRAINER
# --------------------------------------------------

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


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

trainer.fit(
    epochs=EPOCHS,
    patience=PATIENCE,
)