from pathlib import Path

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset


class PlantDiseaseDataset(Dataset):
    """
    PyTorch Dataset for PlantVillage plant disease images.
    """

    def __init__(
        self,
        manifest_path,
        split,
        transform=None,
    ):

        self.manifest_path = Path(manifest_path)

        self.split = split

        self.transform = transform

        # Read the dataset manifest
        self.data = pd.read_csv(
            self.manifest_path
        )

        # Keep only the requested split
        self.data = self.data[
            self.data["split"] == split
        ].reset_index(drop=True)

        if self.data.empty:

            raise ValueError(
                f"No images found for split: {split}"
            )

    def __len__(self):
        """
        Return the number of images.
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Return one image and its class label.
        """

        row = self.data.iloc[index]

        image_path = Path(
            row["image_path"]
        )

        class_id = int(
            row["class_id"]
        )

        # Open image and make sure it is RGB
        image = Image.open(
            image_path
        ).convert("RGB")

        # Apply transformations
        if self.transform is not None:

            image = self.transform(image)

        return image, class_id
        