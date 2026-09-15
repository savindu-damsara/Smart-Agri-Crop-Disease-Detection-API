from pathlib import Path


# ======================================================
# PROJECT ROOT
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ======================================================
# MODEL CONFIGURATION
# ======================================================

MODEL_NAME = "ResNet18"

NUM_CLASSES = 38

IMAGE_SIZE = 224


# ======================================================
# PRODUCTION CHECKPOINT
# ======================================================

MODEL_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "experiments"
    / "experiment_04"
    / "best_model.pt"
)


# ======================================================
# DATA
# ======================================================

DATASET_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_manifest.csv"
)


# ======================================================
# DEVICE
# ======================================================

import torch

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================
# NORMALIZATION
# ======================================================

IMAGE_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGE_STD = [
    0.229,
    0.224,
    0.225
]