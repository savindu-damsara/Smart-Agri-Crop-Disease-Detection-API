from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.dataset import PlantDiseaseDataset
from src.data.transforms import val_transform
from src.models.cnn import PlantDiseaseCNN


# ============================================================
# PROJECT PATHS
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


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "best_model.pt"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32

NUM_CLASSES = 38

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)

print(
    "SMART AGRI - MODEL EVALUATION"
)

print("=" * 70)

print()

print(
    f"Device: {device}"
)

print(
    f"Test batch size: {BATCH_SIZE}"
)

print()


# ============================================================
# LOAD TEST DATASET
# ============================================================

print(
    "Loading test dataset..."
)


test_dataset = PlantDiseaseDataset(
    manifest_path=MANIFEST_PATH,
    split="test",
    transform=val_transform,
)


print(
    f"Test samples: "
    f"{len(test_dataset)}"
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        device.type == "cuda"
    ),
)


# ============================================================
# CREATE MODEL
# ============================================================

print(
    "\nCreating model..."
)


model = PlantDiseaseCNN(
    num_classes=NUM_CLASSES
)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print(
    "Loading best checkpoint..."
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model = model.to(device)

model.eval()


print(
    f"Checkpoint epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Checkpoint validation accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)

print()


# ============================================================
# CLASS NAMES
# ============================================================

import pandas as pd

# Read the manifest to get the class names dynamically
manifest_df = pd.read_csv(MANIFEST_PATH)

# Extract the class names and sort them strictly by class_id
class_mapping = manifest_df[['class_id', 'class_name']].drop_duplicates().sort_values('class_id')
class_names = class_mapping['class_name'].tolist()


print(
    f"Number of classes: "
    f"{len(class_names)}"
)


# ============================================================
# PREDICTION
# ============================================================

all_labels = []

all_predictions = []


print(
    "\nRunning predictions..."
)


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(
            images
        )

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


all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)


print(
    "Prediction completed."
)


# ============================================================
# BASIC ACCURACY
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
)


precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()

print("=" * 70)

print(
    "TEST SET RESULTS"
)

print("=" * 70)

print()

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Macro Precision   : {precision_macro:.4f}"
)

print(
    f"Macro Recall      : {recall_macro:.4f}"
)

print(
    f"Macro F1          : {f1_macro:.4f}"
)

print(
    f"Weighted Precision: {precision_weighted:.4f}"
)

print(
    f"Weighted Recall   : {recall_weighted:.4f}"
)

print(
    f"Weighted F1      : {f1_weighted:.4f}"
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


print()

print("=" * 70)

print(
    "CLASSIFICATION REPORT"
)

print("=" * 70)

print()

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "classification_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "SMART AGRI - "
        "CLASSIFICATION REPORT\n\n"
    )

    file.write(report)


# ============================================================
# PER-CLASS METRICS CSV
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
        zero_division=0,
    )
)


metrics_df = pd.DataFrame({

    "class_name": class_names,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "support": support,

})


metrics_df = metrics_df.sort_values(
    "f1_score"
)


metrics_path = (
    OUTPUT_DIR
    / "per_class_metrics.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
)


plt.figure(
    figsize=(22, 18)
)


sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)


plt.title(
    "Smart Agri - Confusion Matrix"
)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.xticks(
    rotation=90
)


plt.yticks(
    rotation=0
)


plt.tight_layout()


cm_path = (
    OUTPUT_DIR
    / "confusion_matrix.png"
)


plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

cm_normalized = (
    cm.astype(float)
    / cm.sum(
        axis=1,
        keepdims=True
    )
)


cm_normalized = np.nan_to_num(
    cm_normalized
)


plt.figure(
    figsize=(22, 18)
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
    "Smart Agri - Normalized Confusion Matrix"
)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.xticks(
    rotation=90
)


plt.yticks(
    rotation=0
)


plt.tight_layout()


normalized_cm_path = (
    OUTPUT_DIR
    / "normalized_confusion_matrix.png"
)


plt.savefig(
    normalized_cm_path,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "accuracy": accuracy,

    "macro_precision":
        precision_macro,

    "macro_recall":
        recall_macro,

    "macro_f1":
        f1_macro,

    "weighted_precision":
        precision_weighted,

    "weighted_recall":
        recall_weighted,

    "weighted_f1":
        f1_weighted,

}


summary_df = pd.DataFrame(
    [summary]
)


summary_path = (
    OUTPUT_DIR
    / "test_metrics.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print()

print("=" * 70)

print(
    "EVALUATION COMPLETED"
)

print("=" * 70)

print()

print(
    f"Classification report:"
)

print(report_path)

print()

print(
    f"Per-class metrics:"
)

print(metrics_path)

print()

print(
    f"Confusion matrix:"
)

print(cm_path)

print()

print(
    f"Normalized confusion matrix:"
)

print(normalized_cm_path)

print()

print(
    f"Test metrics:"
)

print(summary_path)

print()