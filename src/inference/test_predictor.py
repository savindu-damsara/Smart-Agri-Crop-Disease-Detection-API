from pathlib import Path

import pandas as pd

from .predictor import PlantDiseasePredictor

from src.config import (
    MODEL_CHECKPOINT,
    DATASET_MANIFEST,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = MODEL_CHECKPOINT

MANIFEST_PATH = DATASET_MANIFEST


# --------------------------------------------------
# Select a real test image from the manifest
# --------------------------------------------------

manifest = pd.read_csv(MANIFEST_PATH)

test_image = manifest[
    manifest["split"] == "test"
].iloc[0]["image_path"]

IMAGE_PATH = Path(test_image)


# --------------------------------------------------
# Create predictor
# --------------------------------------------------

predictor = PlantDiseasePredictor(
    checkpoint_path=CHECKPOINT_PATH,
    manifest_path=MANIFEST_PATH,
)


# --------------------------------------------------
# Predict
# --------------------------------------------------

result = predictor.predict(IMAGE_PATH)


# --------------------------------------------------
# Display result
# --------------------------------------------------

print("\n" + "=" * 60)
print("PLANT DISEASE PREDICTION")
print("=" * 60)

print(f"Image      : {IMAGE_PATH}")
print(f"Device     : {predictor.device}")
print(f"Class ID   : {result['class_id']}")
print(f"Disease    : {result['disease']}")
print(f"Confidence : {result['confidence']:.4f}")

print("=" * 60)