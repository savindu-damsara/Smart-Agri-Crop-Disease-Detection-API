from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PlantVillage"
    / "color"
    / "Apple___Apple_scab"
    / "0672ab32-9fce-41f3-ae69-e39c48a0a292___FREC_Scab 3347.JPG"
)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["status"] == "healthy"
    assert data["model"] == "ResNet18"


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "ResNet18"
    assert data["num_classes"] == 38
    assert data["image_size"] == 224

    assert data["test_accuracy"] > 0.99
    assert data["macro_f1"] > 0.99


def test_prediction_endpoint():
    assert TEST_IMAGE.exists(), f"Test image not found: {TEST_IMAGE}"

    with open(TEST_IMAGE, "rb") as image_file:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "test_image.jpg",
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "filename" in data
    assert "prediction" in data

    prediction = data["prediction"]

    assert "class_id" in prediction
    assert "disease" in prediction
    assert "confidence" in prediction

    assert 0 <= prediction["class_id"] < 38
    assert 0.0 <= prediction["confidence"] <= 1.0


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


def test_missing_file():
    response = client.post("/predict")

    assert response.status_code == 422