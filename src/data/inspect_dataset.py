from pathlib import Path
from collections import Counter 

DATASET_PATH = Path("data/raw/PlantVillage/color")

#Configuration

IMAGE_EXTENSTIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

#check dataset directory 

if not DATASET_PATH.exists():
    print(f"Error:Dataset directory not found: {DATASET_PATH}")
    print()
    print("Please check the dataset path")


#find class directories 

class_directories = sorted(
    [
        directory 
        for directory in DATASET_PATH.iterdir()
        if directory.is_dir()
    ]
)

print("="* 60)
print("SMART AGRI - DATASET INSPECTION")
print("="* 60)

print(f"\nDataset path :")
print(DATASET_PATH.resolve())

print(f"\nNumber if class folders :")
print(len(class_directories))

print("\nClasses")
print("-"* 60)

for index, directory in enumerate(class_directories, start=1):
    print(f"{index:02d}. {directory.name}")

#Count images in every class 

print("\n"+"="*60)
print("Image Count")
print("="*60)

class_counts = {}

total_images = 0

for directory in class_directories:

    image_files = [
        file 
        for file in directory.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSTIONS
    ]

    count = len(image_files)

    class_counts[directory.name] = count

    total_images += count

    print(f"{directory.name:<55} {count}")

#Dataset Summary

print("\n" + "="*60)
print("Dataset Summary")
print("=" * 60)

print(f"Number of classes:  {len(class_directories)}")
print(f"Total images : {total_images}")

if class_counts:

    smallest_class = min(
        class_counts,
        key=class_counts.get
    )

    largest_class = max(
        class_counts,
        key=class_counts.get
    )

    print(
        f"Smallest class : "
        f"{smallest_class} ({class_counts[smallest_class]})"

    )

    print(
        f"Largest class :"
        f"{largest_class} ({class_counts[largest_class]})"
    )

print("\nDataset inspection completed.")