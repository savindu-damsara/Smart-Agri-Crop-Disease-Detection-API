from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import PlantDiseaseDataset
from src.models.resnet18_transfer import PlantDiseaseResNet18


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_manifest.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "experiments"
    / "experiment_04"
    / "best_model.pt"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
    / "experiment_04"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32
NUM_CLASSES = 38
NUM_WORKERS = 0
IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = PlantDiseaseDataset(
    manifest_path=MANIFEST_PATH,
    split="test",
    transform=test_transform,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


print(f"Test samples: {len(test_dataset):,}")


# ============================================================
# LOAD MODEL
# ============================================================

model = PlantDiseaseResNet18(
    num_classes=NUM_CLASSES
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


print(
    f"Loaded checkpoint from epoch "
    f"{checkpoint['epoch']}"
)

print(
    f"Checkpoint validation loss: "
    f"{checkpoint['val_loss']:.4f}"
)

print(
    f"Checkpoint validation accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = sorted(
    test_dataset.data["class_name"].unique()
)

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"found {len(class_names)}"
    )


# ============================================================
# PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
)


weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )
)


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test Accuracy       : {accuracy:.4f}"
)

print(
    f"Macro Precision     : {macro_precision:.4f}"
)

print(
    f"Macro Recall        : {macro_recall:.4f}"
)

print(
    f"Macro F1            : {macro_f1:.4f}"
)

print(
    f"Weighted Precision  : {weighted_precision:.4f}"
)

print(
    f"Weighted Recall     : {weighted_recall:.4f}"
)

print(
    f"Weighted F1         : {weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0,
)


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)


report_path = (
    REPORT_DIR
    / "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# PER-CLASS METRICS
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
)


per_class_df = pd.DataFrame(
    {
        "class_id": range(NUM_CLASSES),
        "class_name": class_names,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support,
    }
)


per_class_path = (
    REPORT_DIR
    / "per_class_metrics.csv"
)

per_class_df.to_csv(
    per_class_path,
    index=False
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
)


plt.figure(
    figsize=(20, 17)
)

sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)

plt.title(
    "ResNet18 Test Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.xticks(
    rotation=90
)

plt.yticks(
    rotation=0
)

plt.tight_layout()


cm_path = (
    REPORT_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

cm_normalized = (
    cm.astype("float")
    / cm.sum(
        axis=1,
        keepdims=True
    )
)


cm_normalized = np.nan_to_num(
    cm_normalized
)


plt.figure(
    figsize=(20, 17)
)

sns.heatmap(
    cm_normalized,
    annot=False,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    vmin=0,
    vmax=1,
)

plt.title(
    "ResNet18 Normalized Test Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.xticks(
    rotation=90
)

plt.yticks(
    rotation=0
)

plt.tight_layout()


normalized_cm_path = (
    REPORT_DIR
    / "normalized_confusion_matrix.png"
)

plt.savefig(
    normalized_cm_path,
    dpi=300
)

plt.close()


# ============================================================
# SAVE OVERALL METRICS
# ============================================================

metrics_df = pd.DataFrame(
    [
        {
            "model": "ResNet18 Transfer Learning",
            "checkpoint_epoch": checkpoint["epoch"],
            "validation_loss": checkpoint["val_loss"],
            "validation_accuracy": checkpoint["val_accuracy"],
            "test_accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
        }
    ]
)


metrics_path = (
    REPORT_DIR
    / "test_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print(
    f"Reports saved to: {REPORT_DIR}"
)