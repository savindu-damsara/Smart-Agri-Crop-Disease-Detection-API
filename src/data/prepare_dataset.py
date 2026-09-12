from pathlib import Path
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split


# Configuration


DATASET_PATH = Path("data/raw/PlantVillage/color")

OUTPUT_PATH = Path("data/processed")

RANDOM_STATE = 42

TRAIN_RATIO = 0.80 
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# Helper Functions


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate a SHA-256 hash for an image file.

    This allows us to identify exact duplicate files.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:

        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


# Validate Configuration


if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset directory not found:\n"
        f"{DATASET_PATH.resolve()}"
    )


if TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO != 1.0:

    raise ValueError(
        "Train, validation and test ratios must add up to 1.0."
    )


# Find Images


print("=" * 70)
print("SMART AGRI - DATASET PREPARATION")
print("=" * 70)

print("\nScanning dataset...")

records = []


class_directories = sorted(
    [
        directory
        for directory in DATASET_PATH.iterdir()
        if directory.is_dir()
    ]
)


if len(class_directories) != 38:

    print(
        f"\nWARNING: Expected 38 classes, "
        f"but found {len(class_directories)}."
    )


# Build Image Records


for class_directory in class_directories:

    class_name = class_directory.name

    for image_path in class_directory.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        records.append(
            {
                "image_path": str(image_path),
                "class_name": class_name,
            }
        )


dataset_df = pd.DataFrame(records)


if dataset_df.empty:

    raise RuntimeError(
        "No images were found in the dataset."
    )


print(f"\nImages found: {len(dataset_df):,}")


# Create Class IDs


class_names = sorted(
    dataset_df["class_name"].unique()
)

class_to_id = {
    class_name: class_id
    for class_id, class_name
    in enumerate(class_names)
}

dataset_df["class_id"] = dataset_df["class_name"].map(
    class_to_id
)


print(f"Classes found: {len(class_names)}")


# Check Exact Duplicate Images


print("\nChecking for exact duplicate files...")

hashes = []

for index, image_path in enumerate(
    dataset_df["image_path"],
    start=1
):

    file_hash = calculate_file_hash(
        Path(image_path)
    )

    hashes.append(file_hash)

    if index % 5000 == 0:

        print(
            f"Processed {index:,} images..."
        )


dataset_df["file_hash"] = hashes


duplicate_mask = dataset_df.duplicated(
    subset="file_hash",
    keep=False
)

duplicate_count = duplicate_mask.sum()


print(
    f"\nExact duplicate image records: "
    f"{duplicate_count:,}"
)


# Check Duplicate Label Conflicts

duplicate_groups = (
    dataset_df[duplicate_mask]
    .groupby("file_hash")["class_name"]
    .nunique()
)

conflicting_duplicates = duplicate_groups[
    duplicate_groups > 1
]


if len(conflicting_duplicates) > 0:

    raise RuntimeError(
        "The same image content appears with different "
        "class labels. This must be investigated before "
        "creating train/validation/test splits."
    )

# Remove Exact Duplicates


before_deduplication = len(dataset_df)

dataset_df = dataset_df.drop_duplicates(
    subset="file_hash",
    keep="first"
).reset_index(drop=True)


after_deduplication = len(dataset_df)

removed_duplicates = (
    before_deduplication -
    after_deduplication
)


print(
    f"Duplicate records removed: "
    f"{removed_duplicates:,}"
)

print(
    f"Unique images available: "
    f"{after_deduplication:,}"
)

# First Split: Train vs Temporary


train_df, temporary_df = train_test_split(
    dataset_df,
    test_size=VALIDATION_RATIO + TEST_RATIO,
    random_state=RANDOM_STATE,
    stratify=dataset_df["class_id"],
)

# Second Split: Validation vs Test


relative_test_ratio = (
    TEST_RATIO /
    (VALIDATION_RATIO + TEST_RATIO)
)


validation_df, test_df = train_test_split(
    temporary_df,
    test_size=relative_test_ratio,
    random_state=RANDOM_STATE,
    stratify=temporary_df["class_id"],
)


# Add Split Column


train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
validation_df["split"] = "validation"
test_df["split"] = "test"


# Combine

final_df = pd.concat(
    [
        train_df,
        validation_df,
        test_df
    ],
    ignore_index=True
)


# Sort for Reproducibility

final_df = final_df.sort_values(
    by=["split", "class_id", "image_path"]
).reset_index(drop=True)

# Save Manifest

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


manifest_path = OUTPUT_PATH / "dataset_manifest.csv"

final_df.to_csv(
    manifest_path,
    index=False
)

# Save Class Mapping

class_mapping_df = pd.DataFrame(
    {
        "class_id": list(class_to_id.values()),
        "class_name": list(class_to_id.keys()),
    }
)

class_mapping_path = OUTPUT_PATH / "class_mapping.csv"

class_mapping_df.to_csv(
    class_mapping_path,
    index=False
)

# Print Split Summary

print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

print(
    f"Training images   : {len(train_df):,}"
)

print(
    f"Validation images : {len(validation_df):,}"
)

print(
    f"Test images       : {len(test_df):,}"
)

print(
    f"Total images      : {len(final_df):,}"
)


# Verify Class Distribution

print("\nClass distribution by split:")

split_distribution = pd.crosstab(
    final_df["class_name"],
    final_df["split"]
)

print(split_distribution)



# Final Information


print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETED")
print("=" * 70)

print(
    f"\nManifest saved to:\n"
    f"{manifest_path.resolve()}"
)

print(
    f"\nClass mapping saved to:\n"
    f"{class_mapping_path.resolve()}"
)