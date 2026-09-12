import pandas as pd
import torch


def create_class_weights(
    manifest_path,
    num_classes,
):
    """
    Create class weights from the training split.

    Uses inverse square-root frequency so that
    very rare classes receive more importance
    without creating excessively large weights.
    """

    df = pd.read_csv(
        manifest_path
    )


    train_df = df[
        df["split"] == "train"
    ].copy()


    # Count labels
    class_counts = (
        train_df["class_id"]
        .value_counts()
        .sort_index()
    )


    counts = torch.tensor(
        [
            class_counts.get(
                class_id,
                0
            )
            for class_id in range(
                num_classes
            )
        ],
        dtype=torch.float32,
    )


    # Avoid division by zero
    counts = torch.clamp(
        counts,
        min=1.0
    )


    # Inverse square-root weighting
    weights = 1.0 / torch.sqrt(
        counts
    )


    # Normalize around mean 1
    weights = (
        weights
        / weights.mean()
    )


    return weights