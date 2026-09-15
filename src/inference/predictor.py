from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.config import (
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
)

from src.models.resnet18_transfer import (
    PlantDiseaseResNet18
)


class PlantDiseasePredictor:

    def __init__(
        self,
        checkpoint_path,
        manifest_path,
        device=None
    ):

        self.checkpoint_path = Path(checkpoint_path)
        self.manifest_path = Path(manifest_path)

        # --------------------------------------------------
        # Select device
        # --------------------------------------------------

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        # --------------------------------------------------
        # Load class names
        # --------------------------------------------------

        self.class_names = self._load_class_names()

        # --------------------------------------------------
        # Load trained model
        # --------------------------------------------------

        self.model = self._load_model()

        # --------------------------------------------------
        # Image preprocessing
        # --------------------------------------------------

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # ======================================================
    # LOAD CLASS NAMES
    # ======================================================

    def _load_class_names(self):

        df = pd.read_csv(
            self.manifest_path
        )

        class_mapping = (
            df[
                ["class_id", "class_name"]
            ]
            .drop_duplicates()
            .sort_values("class_id")
        )

        class_names = dict(
            zip(
                class_mapping["class_id"],
                class_mapping["class_name"]
            )
        )

        return class_names

    # ======================================================
    # LOAD MODEL
    # ======================================================

    def _load_model(self):

        # Create the SAME architecture used during training
        model = PlantDiseaseResNet18(
            num_classes=len(self.class_names)
        )

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device
        )

        # Load trained weights
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Move model to CPU/GPU
        model.to(self.device)

        # Evaluation mode
        model.eval()

        return model

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(self, image_path):

        image_path = Path(image_path)

        # Open image
        image = Image.open(
            image_path
        ).convert("RGB")

        # Apply preprocessing
        image_tensor = self.transform(
            image
        )

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Move image to same device as model
        image_tensor = image_tensor.to(
            self.device
        )

        # Disable gradient calculation
        with torch.no_grad():

            # Model prediction
            outputs = self.model(
                image_tensor
            )

            # Convert logits to probabilities
            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            # Get highest probability
            confidence, predicted_class = torch.max(
                probabilities,
                dim=1
            )

        # Convert tensors to Python values
        class_id = predicted_class.item()

        confidence_value = confidence.item()

        # Convert class ID into disease name
        disease_name = self.class_names[
            class_id
        ]

        # Return clean result
        return {
            "class_id": class_id,
            "disease": disease_name,
            "confidence": confidence_value
        }