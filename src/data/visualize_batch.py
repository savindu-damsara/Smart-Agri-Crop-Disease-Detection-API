from pathlib import Path
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader

from dataset import PlantDiseaseDataset
from transforms import train_transform


MANIFEST_PATH = Path(
    "data/processed/dataset_manifest.csv"
)


dataset = PlantDiseaseDataset(
    manifest_path=MANIFEST_PATH,
    split="train",
    transform=train_transform,
)


loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)


images, labels = next(
    iter(loader)
)


# ImageNet normalization values
mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


# Undo normalization for visualization
images = images * std + mean

images = images.clamp(
    0,
    1
)


fig, axes = plt.subplots(
    2,
    4,
    figsize=(12, 6)
)


for ax, image, label in zip(
    axes.flatten(),
    images,
    labels
):

    image = image.permute(
        1,
        2,
        0
    )

    ax.imshow(image)

    ax.set_title(
        f"Class ID: {label.item()}"
    )

    ax.axis("off")


plt.tight_layout()

plt.show()