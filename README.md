# Smart Agri Crop Disease Detection API

An end-to-end deep learning application for detecting plant/crop
diseases from leaf images. the project uses PyTorch with a trained **ResNet18
CNN** , a **FastAPI REST API**, and a **Streamlit user interface** so
that a user can upload a crop image and receive a predicted disease
class with confidence.

> **Project status:** Working end-to-end. The API, Streamlit interface,
> trained PyTorch model, prediction pipeline, automated API tests, Docker setup,
> and CI test workflow have been implemented and tested.

------------------------------------------------------------------------

## 1. What This Project Does

The system follows this flow:

``` text
Leaf Image
    ↓
Streamlit Web Interface
    ↓
FastAPI /predict endpoint
    ↓
Image validation
    ↓
Preprocessing / transformation
    ↓
Trained ResNet18 model
    ↓
Disease class prediction
    ↓
Confidence score
    ↓
Prediction displayed in Streamlit
```

The project is designed as a complete ML application rather than only a
Jupyter Notebook model.

------------------------------------------------------------------------

# 2. Main Features

-   Crop/plant disease classification from leaf images
-   Deep learning image-classification model using **ResNet18**
-   Image preprocessing and transformation pipeline
-   FastAPI REST API
-   Interactive API documentation through Swagger/OpenAPI
-   Streamlit web interface
-   Prediction confidence output
-   Input validation for:
    -   JPG
    -   JPEG
    -   PNG
    -   WEBP
-   Maximum upload size validation: **5 MB**
-   Actual image-content validation using Pillow
-   Temporary upload handling with automatic cleanup
-   Health-check endpoint
-   Model-information endpoint
-   Automated API tests using pytest
-   Dockerized API and Streamlit services
-   Docker Compose orchestration
-   GitHub Actions CI test workflow
-   Separate handling of source code, model artifacts, and dataset files

------------------------------------------------------------------------

# 3. Technologies Used

## Machine Learning / Deep Learning

-   Python 3.11
-   PyTorch
-   Torchvision
-   ResNet18
-   Pillow (PIL)
-   NumPy
-   Pandas
-   Scikit-learn
-   Matplotlib
-   Jupyter Notebook

## Backend / API

-   FastAPI
-   Uvicorn
-   Pydantic
-   Python Multipart
-   HTTPX

## Frontend / User Interface

-   Streamlit
-   Requests

## Testing

-   Pytest
-   FastAPI TestClient
-   HTTPX

## Deployment / Containerization

-   Docker
-   Docker Compose
-   Python slim Docker images

## Development / Version Control

-   VS Code
-   Git
-   GitHub
-   GitHub Actions
-   Python virtual environment (`venv`)

------------------------------------------------------------------------

# 4. Project Structure

The important project structure is approximately:

``` text
Smart-Agri-Crop-Disease-Detection-API/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── data/
│   ├── ... dataset files ...
│   └── uploads/
│
├── models/
│   └── ... trained model/checkpoint files ...
│
├── notebooks/
│   └── ... experimentation and training notebooks ...
│
├── report/
│   └── outputs.txt
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── transforms.py
│   │
│   ├── inference/
│   │   └── predictor.py
│   │
│   ├── models/
│   │   └── cnn.py
│   │
│   ├── training/
│   │   └── ...
│   │
│   └── config.py
│
├── streamlit_app/
│   └── ...
│
├── tests/
│   └── test_api.py
│
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── pytest.ini
├── requirements-api.txt
├── requirements-ui.txt
├── requirements.txt
└── README.md
```

Large model and dataset files may be distributed separately from the Git
repository because Git repositories are not a suitable place for large
ML artifacts.

------------------------------------------------------------------------

# 5. Dataset and Model Files

The repository contains the source code and configuration required to
run the project, while the large model/data artifacts can be downloaded
separately.

### Google Drive model + dataset archive

The trained model files and dataset are provided in the following Google
Drive folder:

https://drive.google.com/drive/folders/1_ybp4_NdIG4PhDf8B4rufyRAQcWoBD73?usp=sharing

Download the required archive and extract it into the **root directory
of the cloned project**.

After extraction, the project should contain the required:

``` text
models/
data/
```

directories.

### Important

Do not rename the model files or change the expected directory structure
unless the paths are also updated in the project configuration.

The model checkpoint and dataset manifest paths are configured through
the project's configuration module.

------------------------------------------------------------------------

# 6. How the Machine Learning Work Was Done

The machine learning component was developed as an experimental workflow
before integrating the final model into the API.

## Step 1 --- Dataset preparation

The image dataset was collected and organized into disease classes.

The data pipeline was created so that images could be:

1.  Loaded from the dataset
2.  Associated with their class labels
3.  Transformed into the format required by the CNN
4.  Loaded efficiently using PyTorch data loaders
5.  Used consistently during training and evaluation

------------------------------------------------------------------------

## Step 2 --- Image preprocessing

Because CNN models require a consistent input format, the images were
transformed before being passed to the network.

The preprocessing pipeline handles operations such as:

-   Resizing images
-   Converting images into tensors
-   Normalization
-   Training-time transformations/augmentation where applicable
-   Validation/test transformations

The final model uses an input image size of:

``` text
224 × 224 pixels
```

------------------------------------------------------------------------

# 7. Model Experimentation

The final ResNet18 model was not selected simply by creating one model
and assuming that it was suitable.

The development process involved experimentation with the
image-classification pipeline and model configurations.

The general experiment cycle was:

``` text
Prepare dataset
      ↓
Create preprocessing pipeline
      ↓
Train candidate model/configuration
      ↓
Evaluate on validation/test data
      ↓
Record results
      ↓
Compare experiments
      ↓
Select final model
      ↓
Retrain/save final model
      ↓
Integrate into inference API
```

During experimentation, the important goal was to determine which
configuration provided a strong balance between:

-   Classification performance
-   Generalization to unseen images
-   Model complexity
-   Training practicality
-   Inference practicality
-   Ease of integration into the final API

The training and evaluation outputs from the experiments were recorded
in:

``` text
report/outputs.txt
```

This file is the detailed experiment record and should be consulted for
the individual training runs, metrics, and outputs rather than relying
only on the final API response.

------------------------------------------------------------------------

# 8. Why ResNet18 Was Used

After the experimentation stage, **ResNet18** was selected as the final
CNN architecture used by the application.

ResNet18 is a convolutional neural network based on residual
connections. Residual connections help the network learn deeper
representations while making optimization easier than a plain deep CNN.

For this project, ResNet18 provided a practical architecture for:

-   Leaf-image classification
-   Strong classification performance
-   Reasonable model size
-   Practical inference
-   Integration into a local/containerized API

The final application therefore uses:

``` text
Model: ResNet18
Input: 224 × 224 image
Output: Disease class + confidence
```

------------------------------------------------------------------------

# 9. Training Process

The selected model was trained using the prepared dataset and PyTorch
training pipeline.

The training workflow consisted of:

1.  Loading the training data
2.  Applying training transformations
3.  Loading batches through a DataLoader
4.  Passing images through ResNet18
5.  Calculating classification loss
6.  Performing backpropagation
7.  Updating model parameters
8.  Evaluating model performance
9.  Saving the trained checkpoint

The trained checkpoint is then loaded by the inference component rather
than retraining the model every time the API starts.

------------------------------------------------------------------------

# 10. Evaluation

The trained model was evaluated using classification metrics.

The final model information exposed by the API records:

``` text
Test Accuracy: 99.63%
Macro F1:      99.38%
```

The API exposes these values through:

``` text
GET /model-info
```

The experiment outputs, including the training/evaluation runs used
during development, are preserved in:

``` text
report/outputs.txt
```

### Why Macro F1 was also considered

Accuracy alone can hide poor performance on individual classes,
especially when class distributions are not perfectly balanced.

Macro F1 calculates the F1 score for each class and then gives equal
importance to the classes when calculating the overall score.

Therefore, using both accuracy and macro F1 gives a more useful view of
classification performance.

------------------------------------------------------------------------

# 11. From Trained Model to Real Application

After the final model was selected and evaluated, it was integrated into
a reusable prediction class.

The inference component is responsible for:

1.  Loading the saved model checkpoint
2.  Loading the class information/manifest
3.  Preparing an input image
4.  Applying the required transformations
5.  Running the model in inference mode
6.  Obtaining the predicted class
7.  Calculating the prediction confidence
8.  Returning structured prediction information

The API does not retrain the model.

It uses the already-trained checkpoint for inference.

------------------------------------------------------------------------

# 12. FastAPI Backend

FastAPI was used to expose the ML model as a REST API.

The main endpoints are:

### `GET /`

Checks whether the API is running.

### `GET /health`

Returns the API health status, model name, and device being used.

### `GET /model-info`

Returns information such as:

-   Model name
-   Number of classes
-   Input image size
-   Test accuracy
-   Macro F1

### `POST /predict`

Accepts an uploaded image and returns the predicted disease.

Example response structure:

``` json
{
  "success": true,
  "filename": "leaf.jpg",
  "prediction": {
    "class_id": 0,
    "disease": "Example Disease",
    "confidence": 0.98
  }
}
```

------------------------------------------------------------------------

# 13. API Input Validation

The `/predict` endpoint includes several validation layers.

### File existence

A request without a valid filename is rejected.

### File extension

Only the following extensions are accepted:

``` text
.jpg
.jpeg
.png
.webp
```

### File size

The maximum accepted image size is:

``` text
5 MB
```

### Actual image validation

The file is opened and verified using Pillow.

This prevents a file with a valid image extension but invalid contents
from being sent directly to the model.

### Temporary file cleanup

Uploaded images are temporarily stored for prediction and then removed
using cleanup logic.

This prevents unnecessary accumulation of uploaded images.

------------------------------------------------------------------------

# 14. Streamlit Interface

A Streamlit interface was created on top of the FastAPI service.

The user can:

1.  Open the Streamlit application
2.  Upload a crop/leaf image
3.  Submit the image for prediction
4.  Send the image to the FastAPI backend
5.  Receive the prediction
6.  View the predicted disease and confidence

This provides a simple graphical interface without requiring the user to
interact directly with the API.

------------------------------------------------------------------------

# 15. API Documentation

FastAPI automatically provides interactive API documentation.

When the API is running, Swagger UI can be accessed through:

``` text
http://localhost:8000/docs
```

This allows the API endpoints to be tested directly from a browser.

The OpenAPI documentation is useful for demonstrating the backend
independently from the Streamlit interface.

------------------------------------------------------------------------

# 16. Testing

Automated tests were created using **pytest**.

The API test suite covers:

``` text
test_root_endpoint
test_health_endpoint
test_model_info_endpoint
test_prediction_endpoint
test_invalid_file_type
test_missing_file
```

The final local test run produced:

``` text
6 passed
```

Example:

``` text
tests/test_api.py::test_root_endpoint PASSED
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_model_info_endpoint PASSED
tests/test_api.py::test_prediction_endpoint PASSED
tests/test_api.py::test_invalid_file_type PASSED
tests/test_api.py::test_missing_file PASSED

6 passed, 2 warnings
```

The warnings were dependency deprecation warnings and did not cause the
tests to fail.

------------------------------------------------------------------------

# 17. Docker

The project was containerized so that the API and Streamlit application
can run consistently without requiring the complete local Python
environment.

Separate Dockerfiles are used for the services:

``` text
Dockerfile.api
Dockerfile.streamlit
```

Docker Compose is used to manage the services together.

Typical commands:

``` bash
docker compose build
```

Then:

``` bash
docker compose up
```

To stop the services:

``` bash
docker compose down
```

The Docker build and Compose setup were successfully tested during
development, and both the API documentation and Streamlit interface were
verified as working.

------------------------------------------------------------------------

# 18. GitHub Actions CI

A GitHub Actions workflow was added to automatically run the test suite.

The workflow:

1.  Checks out the repository
2.  Sets up Python 3.11
3.  Installs API dependencies
4.  Installs pytest and HTTPX
5.  Runs:

``` bash
pytest -v
```

This provides an automated check that the API test suite continues to
pass when changes are pushed to the repository or submitted through a
pull request.

------------------------------------------------------------------------

# 19. How to Run the Project Locally

## Prerequisites

Install:

-   Git
-   Python 3.11
-   VS Code
-   Docker Desktop (recommended for the containerized setup)

------------------------------------------------------------------------

## Option A --- Run with Docker

Clone the repository:

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Enter the project:

``` bash
cd Smart-Agri-Crop-Disease-Detection-API
```

Download the model/data archive from the Google Drive link provided
above.

Extract the required `models/` and `data/` directories into the project
root.

Then build:

``` bash
docker compose build
```

Start:

``` bash
docker compose up
```

Open the Streamlit application using the port shown by Docker Compose.

Open the API documentation:

``` text
http://localhost:8000/docs
```

------------------------------------------------------------------------

# 20. Option B --- Run Locally with Python

Create a virtual environment:

``` bash
python -m venv venv
```

Activate it on Windows PowerShell:

``` powershell
.\venv\Scripts\Activate.ps1
```

Install the API dependencies:

``` bash
pip install -r requirements-api.txt
```

Install the Streamlit dependencies:

``` bash
pip install -r requirements-ui.txt
```

Make sure the model and dataset files have been downloaded and placed in
the expected directories.

Start the API:

``` bash
uvicorn src.api.main:app --reload
```

Then open:

``` text
http://localhost:8000/docs
```

Start Streamlit in another terminal:

``` bash
streamlit run streamlit_app/app.py
```

Use the URL displayed by Streamlit to open the interface.

------------------------------------------------------------------------

# 21. Running the Tests

From the project root:

``` bash
pytest -v
```

The expected API test result is:

``` text
6 passed
```

If you are running the project with the Docker environment, testing can
also be performed inside the appropriate container.

------------------------------------------------------------------------

# 22. Complete Development Journey

The project was developed from the ML problem through to a usable
application.

### Phase 1 --- Problem definition

The problem was defined as automated crop/plant disease classification
from leaf images.

### Phase 2 --- Dataset preparation

The image dataset was organized into labelled disease classes and
prepared for deep learning.

### Phase 3 --- Data pipeline

A PyTorch dataset/data-loader pipeline and image transformation pipeline
were created.

### Phase 4 --- Experimentation

Different training/model configurations were experimented with. Training
and evaluation results were recorded so that the final model choice was
based on measured performance rather than only assumptions.

The complete recorded experiment output is available in:

``` text
report/outputs.txt
```

### Phase 5 --- Final model selection

ResNet18 was selected as the final architecture after the
experimentation/evaluation stage.

### Phase 6 --- Training

The final model was trained using the prepared dataset and the resulting
checkpoint was saved for inference.

### Phase 7 --- Evaluation

The final model was evaluated using classification metrics. The final
recorded API model information reports:

``` text
Accuracy = 99.63%
Macro F1 = 99.38%
```

### Phase 8 --- Inference pipeline

A prediction component was created to load the trained checkpoint and
convert a new leaf image into a disease prediction and confidence score.

### Phase 9 --- API development

FastAPI was used to expose the model through REST endpoints.

Input validation, file-size restrictions, image verification, temporary
file handling, structured responses, health checks, and model
information were added.

### Phase 10 --- User interface

A Streamlit interface was built so users could upload images and obtain
predictions without manually interacting with the API.

### Phase 11 --- Testing

Automated API tests were written with pytest. The final local test suite
successfully completed with:

``` text
6 passed
```

### Phase 12 --- Containerization

The API and Streamlit application were containerized using Docker and
Docker Compose.

### Phase 13 --- CI

GitHub Actions was configured to automatically install dependencies and
execute the pytest suite.

The result is an end-to-end machine-learning application:

``` text
Dataset
   ↓
Preprocessing
   ↓
Model Experiments
   ↓
Model Evaluation
   ↓
ResNet18 Selection
   ↓
Final Training
   ↓
Saved Model
   ↓
Inference Pipeline
   ↓
FastAPI
   ↓
Streamlit
   ↓
Docker
   ↓
Automated Testing / CI
```

------------------------------------------------------------------------

# 23. Reproducibility Notes

The GitHub repository contains the source code required to understand
and run the application, while large model and dataset artifacts are
distributed separately.

For a complete reproduction:

1.  Clone the repository.
2.  Download the model/data archive from Google Drive.
3.  Extract the required `models/` and `data/` directories into the
    project root.
4.  Install dependencies or use Docker.
5.  Start the API.
6.  Start Streamlit.
7.  Upload a supported leaf image.
8.  Verify the prediction.
9.  Run `pytest -v` to verify the API tests.

The experiment history can be reviewed in:

``` text
report/outputs.txt
```

------------------------------------------------------------------------

# 24. Limitations

-   The model's predictions depend on the quality and distribution of
    the training dataset.
-   Performance on real-world images that differ significantly from the
    training data may vary.
-   The reported evaluation metrics are based on the project's
    evaluation dataset and should not automatically be interpreted as
    real-world field accuracy.
-   The application is intended as a machine-learning project/prototype
    and not as a replacement for professional agricultural diagnosis.

------------------------------------------------------------------------

# 25. Future Improvements

Potential future development includes:

-   More diverse real-world crop images
-   Additional disease classes
-   Improved data augmentation
-   Model optimization for mobile/edge devices
-   Authentication and authorization for the API
-   Production cloud deployment
-   Database integration for prediction history
-   Monitoring of model performance
-   Model versioning
-   Automated model retraining
-   Larger-scale deployment

------------------------------------------------------------------------

# 26. Author / Project

Kannangara Koralalage Don Savindu Damsara 

**Smart Agri Crop Disease Detection API**

Developed as a machine-learning application demonstrating the complete
workflow from dataset preparation and model experimentation to model
evaluation, API integration, UI development, testing, containerization,
and CI.

------------------------------------------------------------------------
