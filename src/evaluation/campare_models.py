from pathlib import Path

import torch

from src.models.cnn import PlantDiseaseCNN
from src.models.cnn_experiment_03 import PlantDiseaseCNNExperiment03
from src.models.resnet18_transfer import PlantDiseaseResNet18


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# MODELS
# ============================================================

models_to_compare = {
    "Baseline CNN": PlantDiseaseCNN(num_classes=38),
    "Experiment 03 CNN": PlantDiseaseCNNExperiment03(num_classes=38),
    "ResNet18": PlantDiseaseResNet18(num_classes=38),
}


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


for name, model in models_to_compare.items():

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\n{name}")
    print("-" * 50)

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )


# ============================================================
# CHECKPOINT SIZES
# ============================================================

checkpoint_paths = {
    "Baseline CNN": (
        PROJECT_ROOT
        / "models"
        / "checkpoints"
        / "best_model.pt"
    ),

    "ResNet18": (
        PROJECT_ROOT
        / "models"
        / "checkpoints"
        / "experiments"
        / "experiment_04"
        / "best_model.pt"
    ),
}


print("\n" + "=" * 70)
print("CHECKPOINT SIZES")
print("=" * 70)


for name, path in checkpoint_paths.items():

    if path.exists():

        size_mb = (
            path.stat().st_size
            / (1024 ** 2)
        )

        print(
            f"{name:<20}: "
            f"{size_mb:.2f} MB"
        )

    else:

        print(
            f"{name:<20}: "
            f"Checkpoint not found"
        )