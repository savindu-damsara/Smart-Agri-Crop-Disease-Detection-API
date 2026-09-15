from pathlib import Path
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from PIL import Image, UnidentifiedImageError

from src.config import (
    MODEL_CHECKPOINT,
    DATASET_MANIFEST,
)

from src.inference.predictor import (
    PlantDiseasePredictor,
)

from src.api.schemas import (
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)


# ======================================================
# CONFIGURATION
# ======================================================

MAX_FILE_SIZE = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ======================================================
# FASTAPI APPLICATION
# ======================================================

app = FastAPI(
    title="Smart Agri Crop Disease Detection API",
    description=(
        "AI-powered crop disease classification "
        "using a ResNet18 deep learning model."
    ),
    version="1.0.0",
)


# ======================================================
# MODEL LOADING
# ======================================================

def create_predictor():
    """
    Create the production ML predictor.

    The model is created only when it is actually needed.
    This prevents the large model checkpoint from being
    loaded when the API module is imported during testing.
    """

    return PlantDiseasePredictor(
        checkpoint_path=MODEL_CHECKPOINT,
        manifest_path=DATASET_MANIFEST,
    )


# Predictor starts as None.
# The real model is loaded lazily when required.
predictor = None


def get_predictor():
    """
    Return the ML predictor.

    If the predictor has not been created yet,
    create it and store it for future requests.
    """

    global predictor

    if predictor is None:
        predictor = create_predictor()

    return predictor


# ======================================================
# ROOT
# ======================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": (
            "Smart Agri Crop Disease Detection API"
        ),
        "status": "running",
    }


# ======================================================
# HEALTH
# ======================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health():

    current_predictor = get_predictor()

    return {
        "success": True,
        "status": "healthy",
        "model": "ResNet18",
        "device": str(current_predictor.device),
    }


# ======================================================
# MODEL INFO
# ======================================================

@app.get(
    "/model-info",
    response_model=ModelInfoResponse
)
def model_info():

    current_predictor = get_predictor()

    return {
        "model": "ResNet18",
        "num_classes": len(
            current_predictor.class_names
        ),
        "image_size": 224,
        "test_accuracy": 0.9963,
        "macro_f1": 0.9938,
    }


# ======================================================
# PREDICTION
# ======================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------
    # Validate filename
    # --------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file was provided.",
        )

    # --------------------------------------------------
    # Validate extension
    # --------------------------------------------------

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Allowed formats: JPG, JPEG, PNG, WEBP."
            ),
        )

    # --------------------------------------------------
    # Read file
    # --------------------------------------------------

    contents = await file.read()

    # --------------------------------------------------
    # Validate file size
    # --------------------------------------------------

    if len(contents) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "Image is too large. "
                "Maximum allowed size is 5 MB."
            ),
        )

    # --------------------------------------------------
    # Temporary upload directory
    # --------------------------------------------------

    upload_directory = (
        Path("data") / "uploads"
    )

    upload_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Generate safe temporary filename
    # --------------------------------------------------

    temporary_filename = (
        f"{uuid4().hex}{extension}"
    )

    upload_path = (
        upload_directory
        / temporary_filename
    )

    # --------------------------------------------------
    # Save file temporarily
    # --------------------------------------------------

    with open(
        upload_path,
        "wb"
    ) as buffer:

        buffer.write(contents)

    # --------------------------------------------------
    # Validate that the file is actually an image
    # --------------------------------------------------

    try:

        with Image.open(
            upload_path
        ) as image:

            image.verify()

    except (
        UnidentifiedImageError,
        OSError,
    ):

        if upload_path.exists():
            upload_path.unlink()

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    # --------------------------------------------------
    # Run model prediction
    # --------------------------------------------------

    try:

        # Load the real predictor only when prediction
        # is actually requested.
        current_predictor = get_predictor()

        result = current_predictor.predict(
            upload_path
        )

    except Exception as error:

        if upload_path.exists():
            upload_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {error}"
            ),
        )

    finally:

        # Always remove temporary image
        if upload_path.exists():
            upload_path.unlink()

    # --------------------------------------------------
    # Return structured response
    # --------------------------------------------------

    return {
        "success": True,
        "filename": file.filename,
        "prediction": {
            "class_id": result["class_id"],
            "disease": result["disease"],
            "confidence": result["confidence"],
        },
    }