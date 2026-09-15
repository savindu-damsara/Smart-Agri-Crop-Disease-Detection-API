import torch.nn as nn
from torchvision import models


class PlantDiseaseResNet18(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()

        # Load ResNet18 pretrained on ImageNet
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Get number of inputs to the original classifier
        num_features = self.model.fc.in_features

        # Replace the original ImageNet classifier
        self.model.fc = nn.Linear(
            num_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)