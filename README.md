# Smart Agri Crop Disease Detection API

An end-to-end deep learning application for crop disease classification from leaf images. The project combines a trained **ResNet18** image-classification model with a **FastAPI REST API** and a **Streamlit web interface**, with Docker-based execution and automated API testing.

> **Note:** The trained model checkpoint and dataset are kept outside the Git repository because of their size. A downloadable ZIP containing the required `models/` and `data/` directories is provided through Google Drive.

---

## Features

- Crop/plant disease classification from uploaded leaf images
- ResNet18 deep learning model
- 224 × 224 image preprocessing
- Confidence score returned with each prediction
- FastAPI REST API
- Interactive Swagger API documentation
- Streamlit web interface
- Image type validation: JPG, JPEG, PNG, WEBP
- Maximum upload size validation: 5 MB
- Temporary upload handling with automatic cleanup
- Docker and Docker Compose support
- Automated API tests with pytest
- GitHub Actions CI workflow
- Model information and health-check endpoints

---

## Technology Stack

### Machine Learning / Deep Learning
- Python
- PyTorch
- Torchvision
- ResNet18
- NumPy
- Pandas
- Pillow (PIL)

### Backend / API
- FastAPI
- Uvicorn
- Pydantic
- REST API
- Python `UploadFile`
- Swagger / OpenAPI

### Frontend
- Streamlit
- Requests

### Testing
- Pytest
- FastAPI TestClient
- HTTPX

### DevOps / Deployment
- Docker
- Docker Compose
- GitHub Actions
- Linux-based CI environment

### Development Tools
- Git
- GitHub
- Virtual environment (`venv`)
- PowerShell / Windows

---

## Project Architecture

```text
                    ┌──────────────────────┐
                    │   User / Browser     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Streamlit UI      │
                    │  Image Upload + UI    │
                    └──────────┬───────────┘
                               │ HTTP
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI         │
                    │      REST API        │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Image Validation      │
                    │ • Extension          │
                    │ • File Size           │
                    │ • Image Integrity     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ PlantDiseasePredictor│
                    │   ResNet18 Model     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Prediction Result    │
                    │ Disease + Confidence │
                    └──────────────────────┘
```

---

## Project Structure

After downloading the model/data package, the expected structure is approximately:

```text
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
│   ├── ... trained model checkpoint ...
│   └── ...
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── config.py
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
│   └── training/
│       └── ...
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
├── .gitignore
└── README.md
```

---

# 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Smart-Agri-Crop-Disease-Detection-API
```

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again.

---

# 2. Download the Model and Dataset

The trained model and dataset are distributed separately from GitHub.

### Google Drive

[Download the model and dataset package](https://drive.google.com/drive/folders/1_ybp4_NdIG4PhDf8B4rufyRAQcWoBD73?usp=sharing)

Download the ZIP containing the `models` and `data` directories.

After downloading:

1. Extract the ZIP.
2. Locate the extracted `models` folder.
3. Locate the extracted `data` folder.
4. Copy both folders into the **root of the cloned repository**.

The final structure should look like:

```text
Smart-Agri-Crop-Disease-Detection-API/
├── data/
├── models/
├── src/
├── tests/
├── streamlit_app/
├── docker-compose.yml
└── README.md
```

### Important

Do **not** put the ZIP file inside the project as a replacement for the folders.

The application expects the actual:

```text
models/
data/
```

directories to exist at the project root.

---

# 3. Install Dependencies

For the API:

```powershell
pip install -r requirements-api.txt
```

For the Streamlit interface:

```powershell
pip install -r requirements-ui.txt
```

For testing:

```powershell
pip install pytest httpx
```

---

# 4. Run the API Locally

From the project root:

```powershell
uvicorn src.api.main:app --reload
```

The API should start locally.

Open:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

ReDoc documentation:

```text
http://127.0.0.1:8000/redoc
```

---

# 5. API Endpoints

## `GET /`

Checks whether the API is running.

Example response:

```json
{
  "success": true,
  "message": "Smart Agri Crop Disease Detection API",
  "status": "running"
}
```

---

## `GET /health`

Returns API/model health information.

Example:

```json
{
  "success": true,
  "status": "healthy",
  "model": "ResNet18",
  "device": "cpu"
}
```

The device may differ depending on the environment.

---

## `GET /model-info`

Returns information about the trained model.

Example:

```json
{
  "model": "ResNet18",
  "num_classes":  ...,
  "image_size": 224,
  "test_accuracy": 0.9963,
  "macro_f1": 0.9938
}
```

---

## `POST /predict`

Accepts a leaf image and returns the predicted disease class.

Supported formats:

```text
.jpg
.jpeg
.png
.webp
```

Maximum file size:

```text
5 MB
```

Example response:

```json
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

The actual disease label and confidence depend on the uploaded image.

---

# 6. Run the Streamlit Application

With the API running, open another terminal.

Activate the virtual environment if necessary:

```powershell
.\venv\Scripts\Activate.ps1
```

Run:

```powershell
streamlit run streamlit_app/app.py
```

The Streamlit interface will provide a browser-based interface for uploading a crop/leaf image and viewing the model prediction.

The Streamlit application communicates with the FastAPI backend rather than directly loading the model in the UI.

---

# 7. Run with Docker Compose

Docker is the recommended way to reproduce the complete application environment.

Make sure Docker Desktop is running.

From the project root:

```powershell
docker compose build
```

Then:

```powershell
docker compose up
```

Or:

```powershell
docker compose up --build
```

The project contains separate services for:

```text
API
Streamlit UI
```

After the containers start, open the ports shown by Docker Compose in the terminal.

To stop the services:

```powershell
docker compose down
```

To rebuild without using cached Docker layers:

```powershell
docker compose build --no-cache
```

---

# 8. Testing

The project includes automated API tests using **pytest**.

Run:

```powershell
pytest -v
```

The test suite verifies:

- Root endpoint
- Health endpoint
- Model information endpoint
- Prediction endpoint
- Invalid file-type handling
- Missing-file handling

A successful run currently produces:

```text
6 passed
```

Example:

```text
tests/test_api.py::test_root_endpoint PASSED
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_model_info_endpoint PASSED
tests/test_api.py::test_prediction_endpoint PASSED
tests/test_api.py::test_invalid_file_type PASSED
tests/test_api.py::test_missing_file PASSED
```

Some environments may display dependency deprecation warnings while tests still pass successfully.

---

# 9. Continuous Integration

GitHub Actions is configured to automatically run the test suite when changes are pushed to `main` or when a pull request targets `main`.

The workflow:

1. Checks out the repository
2. Sets up Python 3.11
3. Installs API dependencies
4. Installs pytest and HTTPX
5. Runs:

```bash
pytest -v
```

This helps verify that API functionality remains intact when code changes are introduced.

---

# 10. Machine Learning Pipeline

The project follows an image-classification workflow:

```text
Raw Dataset
     │
     ▼
Data Preparation
     │
     ▼
Duplicate / Data Quality Checks
     │
     ▼
Dataset Splitting
     │
     ├── Training Set
     ├── Validation Set
     └── Test Set
     │
     ▼
Image Preprocessing
     │
     ▼
Model Training
     │
     ▼
ResNet18
     │
     ▼
Evaluation
     │
     ▼
Best Model Checkpoint
     │
     ▼
FastAPI Inference
```

The project uses a dataset split of approximately:

```text
80% Training
10% Validation
10% Testing
```

Images are processed to the model's expected input size of:

```text
224 × 224
```

---

# 11. Model

The final image-classification model uses **ResNet18**, a convolutional neural network architecture designed for image recognition tasks.

The model receives a preprocessed leaf image and predicts one of the disease classes represented in the training dataset.

The inference pipeline:

```text
Uploaded Image
      │
      ▼
File Validation
      │
      ▼
Image Verification
      │
      ▼
Preprocessing
      │
      ▼
ResNet18
      │
      ▼
Class Probabilities
      │
      ▼
Predicted Class
      │
      ▼
Confidence Score
```

---

# 12. Model Evaluation

The reported final model evaluation includes:

| Metric | Result |
|---|---:|
| Test Accuracy | **99.63%** |
| Macro F1 Score | **99.38%** |
| Input Size | **224 × 224** |
| Architecture | **ResNet18** |

These values describe the evaluation performed on the project's test set. They should not be interpreted as guaranteed performance on real-world photographs that differ from the dataset.

---

# 13. API Input Validation

The prediction endpoint contains multiple validation layers.

### File name validation

The API checks that a filename is provided.

### Extension validation

Only the following extensions are accepted:

```text
JPG
JPEG
PNG
WEBP
```

### File size validation

Uploads larger than 5 MB are rejected.

### Image integrity validation

The uploaded file is opened and verified using Pillow.

This prevents files that merely have an image extension from automatically being treated as valid images.

### Temporary file cleanup

Uploaded files are assigned generated temporary filenames and removed after prediction.

This avoids keeping uploaded user images permanently on the server.

---

# 14. Error Handling

The API returns appropriate HTTP errors for common invalid requests.

Examples include:

```text
400 - Invalid request / unsupported image
413 - Image exceeds 5 MB
500 - Prediction failure
```

This makes the API easier to integrate with other applications and frontends.

---

# 15. Why the Model Is Not Stored Directly in GitHub

The trained model checkpoint and dataset can be large.

GitHub has a 100 MiB limit for individual files in normal Git repositories. Keeping large binary artifacts outside the Git repository avoids unnecessarily bloating Git history and keeps the source repository focused on code and configuration.

The repository therefore contains the code required to load and use the model, while the trained artifacts are distributed separately through Google Drive.

For a reviewer, the workflow is:

```text
Clone GitHub Repository
        │
        ▼
Download Model + Dataset ZIP
        │
        ▼
Extract
        │
        ▼
Copy models/ and data/
to project root
        │
        ▼
Install Dependencies
        │
        ▼
Run Docker Compose
        │
        ▼
Open Streamlit
        │
        ▼
Upload Leaf Image
        │
        ▼
View Disease Prediction
```

---

# 16. Reproducing the Project

A reviewer can reproduce the project using the following short workflow:

```powershell
git clone <YOUR-GITHUB-REPOSITORY-URL>

cd Smart-Agri-Crop-Disease-Detection-API

# Download and extract the model/data ZIP from Google Drive.
# Place models/ and data/ in this directory.

docker compose build

docker compose up
```

Then use the Streamlit URL exposed by Docker Compose.

For API testing:

```powershell
pytest -v
```

For API documentation:

```text
http://localhost:8000/docs
```

---

# 17. Development Without Docker

If Docker is not available, the application can also be run using the Python environment.

### API

```powershell
uvicorn src.api.main:app --reload
```

### Streamlit

In another terminal:

```powershell
streamlit run streamlit_app/app.py
```

### Tests

```powershell
pytest -v
```

---

# 18. Deployment

The project has been containerized using Docker and Docker Compose so that the API and Streamlit services can be reproduced consistently.

A public cloud deployment is **not required to run the project locally**.

The repository is intended to demonstrate:

- Model development
- Model inference
- REST API development
- Frontend integration
- Containerization
- Automated testing
- Continuous integration

---

# 19. Technologies Summary

```text
Python
PyTorch
Torchvision
ResNet18
NumPy
Pandas
Pillow
FastAPI
Uvicorn
Pydantic
Streamlit
Requests
Pytest
HTTPX
Docker
Docker Compose
GitHub Actions
Git
GitHub
```

---

# 20. Project Highlights

- Built an end-to-end **deep learning crop disease classification system**
- Implemented a **ResNet18** image-classification model
- Achieved **99.63% test accuracy** and **99.38% macro F1** on the project test set
- Developed a production-style **FastAPI REST API** for model inference
- Added image validation, file-size restrictions, temporary file handling, and structured API responses
- Built a **Streamlit** interface for interactive predictions
- Containerized the application with **Docker and Docker Compose**
- Added automated API testing with **pytest**
- Added **GitHub Actions CI** for automated test execution

---

# 21. Limitations

- The model's performance depends on how similar new images are to the training data.
- Dataset-based test performance may not represent performance on every real-world crop image.
- The trained model and dataset are distributed separately from the Git repository.
- The current project is designed primarily as a machine-learning application and demonstration rather than a complete agricultural decision-support platform.

---

# 22. Future Improvements

Possible extensions include:

- Larger and more diverse real-world datasets
- Additional crop and disease classes
- Model explainability using Grad-CAM or similar methods
- Confidence-based uncertainty handling
- Image augmentation improvements
- Model optimization for mobile/edge devices
- Authentication and API rate limiting
- Cloud deployment
- Database-backed prediction history
- Monitoring and logging
- Agricultural recommendations based on predicted disease

---

## Author

**Savindu Kannangara**

Data Science Undergraduate — SLIIT

---

## License

Add the project's chosen license here if you decide to publish the repository under an open-source license.

---

## Disclaimer

This project is intended for educational, research, and demonstration purposes. Model predictions should not be treated as a substitute for professional agricultural diagnosis or expert advice.
