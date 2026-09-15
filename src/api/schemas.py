from pydantic import BaseModel


class PredictionResult(BaseModel):
    class_id: int
    disease: str
    confidence: float


class PredictionResponse(BaseModel):
    success: bool
    filename: str
    prediction: PredictionResult


class HealthResponse(BaseModel):
    success: bool
    status: str
    model: str
    device: str


class ModelInfoResponse(BaseModel):
    model: str
    num_classes: int
    image_size: int
    test_accuracy: float
    macro_f1: float