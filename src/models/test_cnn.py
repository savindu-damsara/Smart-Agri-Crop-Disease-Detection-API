import torch

from cnn import PlantDiseaseCNN


# Configuration

BATCH_SIZE = 32
NUM_CLASSES = 38


# Create Model


model = PlantDiseaseCNN(
    num_classes=NUM_CLASSES
)


print("=" * 60)
print("SMART AGRI - CNN MODEL TEST")
print("=" * 60)

print("\nModel created successfully.")

print(
    f"\nNumber of classes: {NUM_CLASSES}"
)


# Create Fake Input


dummy_input = torch.randn(
    BATCH_SIZE,
    3,
    224,
    224,
)


print(
    f"\nInput shape: {dummy_input.shape}"
)


# Forward Pass


output = model(
    dummy_input
)


print(
    f"Output shape: {output.shape}"
)


# Verify Output


expected_shape = (
    BATCH_SIZE,
    NUM_CLASSES
)


assert output.shape == expected_shape


print(
    "\nExpected output shape:",
    expected_shape
)

print(
    "Actual output shape:",
    tuple(output.shape)
)

print("\nCNN forward pass successful!")