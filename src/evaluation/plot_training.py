from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


HISTORY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiments"
    / "training_history.csv"
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
# LOAD HISTORY
# ============================================================

df = pd.read_csv(
    HISTORY_PATH
)


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    df["epoch"],
    df["train_loss"],
    label="Training Loss",
)


plt.plot(
    df["epoch"],
    df["val_loss"],
    label="Validation Loss",
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Training and Validation Loss"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "loss_curve.png",
    dpi=200,
)


plt.close()


# ============================================================
# ACCURACY CURVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    df["epoch"],
    df["train_accuracy"],
    label="Training Accuracy",
)


plt.plot(
    df["epoch"],
    df["val_accuracy"],
    label="Validation Accuracy",
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "Training and Validation Accuracy"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "accuracy_curve.png",
    dpi=200,
)


plt.close()


# ============================================================
# LEARNING RATE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    df["epoch"],
    df["learning_rate"],
    marker="o",
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Learning Rate"
)


plt.title(
    "Learning Rate Schedule"
)


plt.yscale(
    "log"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "learning_rate_curve.png",
    dpi=200,
)


plt.close()


print(
    "Training plots generated successfully."
)