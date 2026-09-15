import torch
import torch.nn as nn


class PlantDiseaseCNNExperiment03(nn.Module):
    """
    Experiment 03:
    Deeper CNN with four convolutional blocks.
    """

    def __init__(self, num_classes=38):
        super().__init__()

        # ====================================================
        # Feature Extraction
        # ====================================================

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),


            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),


            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),


            # ------------------------------------------------
            # Block 4 - NEW
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )


        # ====================================================
        # Classification
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256 * 14 * 14,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.5
            ),

            nn.Linear(
                256,
                num_classes,
            ),
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x