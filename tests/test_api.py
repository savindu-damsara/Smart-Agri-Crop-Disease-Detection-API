from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


# ======================================================
# TEST CLIENT
# ======================================================

client = TestClient(app)


# ======================================================
# FAKE PREDICTOR
# ======================================================

class FakePredictor:
    """
    Lightweight predictor used only for API tests.

    It avoids loading the real ResNet18 checkpoint,
    making the tests suitable for GitHub Actions.
    """

    device = "cpu"

    class_names = {
        0: "Apple___Apple_scab",
        1: "Apple___Black_rot",
        2: "Apple___Cedar_apple_rust",
    }

    def predict(self, image_path):
        return {
            "class_id": 0,
            "disease": "Apple___Apple_scab",
            "confidence": 0.95,
        }


# ======================================================
# REPLACE REAL MODEL FOR TESTS
# ======================================================

app.state.predictor = FakePredictor()


# Replace the predictor used by the API endpoints
import src.api.main as main_module

main_module.predictor = FakePredictor()


# ======================================================
# HELPER
# ======================================================

def create_test_image():
    """
    Create a small valid JPEG image in memory.

    No PlantVillage dataset is required.
    """

    image = Image.new(
        "RGB",
        (224, 224),
        color="green",
    )

    image_bytes = BytesIO()

    image.save(
        image_bytes,
        format="JPEG",
    )

    image_bytes.seek(0)

    return image_bytes


# ======================================================
# ROOT ENDPOINT
# ======================================================

def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert data["status"] == "running"


# ======================================================
# HEALTH ENDPOINT
# ======================================================

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert data["status"] == "healthy"

    assert data["model"] == "ResNet18"

    assert data["device"] == "cpu"


# ======================================================
# MODEL INFO ENDPOINT
# ======================================================

def test_model_info_endpoint():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "ResNet18"

    assert data["num_classes"] == 3

    assert data["image_size"] == 224

    assert data["test_accuracy"] > 0.99

    assert data["macro_f1"] > 0.99


# ======================================================
# PREDICTION ENDPOINT
# ======================================================

def test_prediction_endpoint():

    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test_image.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert data["filename"] == "test_image.jpg"

    assert "prediction" in data

    prediction = data["prediction"]

    assert prediction["class_id"] == 0

    assert prediction["disease"] == "Apple___Apple_scab"

    assert prediction["confidence"] == 0.95


# ======================================================
# INVALID FILE TYPE
# ======================================================

def test_invalid_file_type():

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"This is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


# ======================================================
# MISSING FILE
# ======================================================

def test_missing_file():

    response = client.post("/predict")

    assert response.status_code == 422