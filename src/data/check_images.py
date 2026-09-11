from pathlib import Path
from PIL import Image

DATASET_PATH = Path("data/raw/PlantVillage/color")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

total_images = 0
corrupted_images = []

for class_directory in DATASET_PATH.iterdir():

    if not class_directory.is_dir():
        continue

    for image_path in class_directory.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS: 
            continue 

        total_images += 1

        try:

            with Image.open(image_path) as image:
                image.verify()

        except Exception as error:

            corrupted_images.append(
                (image_path, str(error))
            )

print("=" * 60)
print("IMAGE VALIDATION")
print("=" * 60)

print(f"Total images checked : {total_images}")
print(f"Corrupted images     : {len(corrupted_images)}")

if corrupted_images:

    print("\nCorrupted images:")

    for image_path, error in corrupted_images[:20]:

        print(image_path)
        print(error)
        print("-" * 60)

else:

    print("\nAll images passed validation.")