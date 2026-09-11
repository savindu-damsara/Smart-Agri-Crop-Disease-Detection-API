# PlantVillage Dataset — EDA Summary

## Dataset

PlantVillage color image dataset.

## Dataset Configuration

For the main model development pipeline, the color dataset is used.

Grayscale and segmented versions are kept separately for future experiments.

## Classes

The dataset contains 38 disease/healthy classes.

## Analysis Performed

- Number of classes
- Total image count
- Images per class
- Class distribution
- Class imbalance
- Image dimensions
- Image color modes
- Random image inspection
- Pixel value inspection

## Important Observations

The dataset contains multiple plant species and disease categories.

below are the generated dataset summary :

============================================================
SMART AGRI - DATASET SUMMARY
============================================================
Dataset type       : PlantVillage
Dataset version    : Color images
Number of classes  : 38
Total images       : 54,305
Smallest class     : Potato___healthy
Smallest count     : 152
Largest class      : Orange___Haunglongbing_(Citrus_greening)
Largest count      : 5507
Imbalance ratio    : 36.23
============================================================

## EDA Conclusion

The analysis identified the following characteristics:

The dataset contains 38 classes.
The color dataset contains 54,305 images.
Class sizes are highly imbalanced.
The largest class contains 5,507 images.
The smallest class contains 152 images.
The imbalance ratio is approximately 36.23.
Images contain RGB color information.
Image dimensions need to be standardized during preprocessing.
Grayscale and segmented datasets will remain separate from the main
color dataset.
Data leakage and class imbalance must be considered during model
development.

## Next Step

Prepare the dataset for model training using a leakage-safe train/validation/test split and PyTorch transformations.

