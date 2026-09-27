"""
Simple FastAPI Application for ADAS Road Hazard Classifier
- Single-model inference (ResNet-18)
- Input validation and error handling
- Part 1 companion: Interface Layer Implementation
"""

import os
import io
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== Configuration ========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['animal', 'name_board', 'pedestrian', 'pothole', 'road_sign', 'speed_breaker', 'vehicle']
NUM_CLASSES = len(CLASS_NAMES)
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "./models")

# ======================== Model ========================


class CNNModel(nn.Module):
    """ResNet-18 classifier for ADAS road hazard detection"""
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# ======================== Model Loading ========================

def load_model():
    """
    Load trained model from class2/models/v1/.
    Run class1/Part_2_Overfitting_and_Generalization.ipynb first to generate the .pth file.
    Falls back to random weights if file is not found.
    """
    model = CNNModel(NUM_CLASSES).to(DEVICE)

    model_path = os.path.join(LOCAL_MODEL_PATH, 'v1', 'model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        logger.info(f"Loaded model weights from {model_path}")
    else:
        logger.warning(f"Model weights not found at {model_path} — using random weights.")
        logger.warning("Run class1/Part_2_Overfitting_and_Generalization.ipynb to generate model files.")

    model.eval()
    logger.info(f"Model ready on device: {DEVICE}")
    return model


# ======================== API Schemas ========================

class PredictionResponse(BaseModel):
    """Response for model prediction"""
    prediction: str
    confidence: float
    class_probabilities: Dict[str, float]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_version: str
    device: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    version: str
    num_classes: int
    classes: List[str]
    input_shape: str
    device: str


# ======================== App Setup ========================

# Load model
model = load_model()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create FastAPI app
app = FastAPI(
    title="ADAS Classifier API",
    description="CNN model for detecting ADAS road hazards",
    version="1.0.0"
)

logger.info("FastAPI application initialized")


# ======================== Utility Functions ========================

def validate_image_file(file_ext: str) -> bool:
    """Validate file extension"""
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    return file_ext.lower() in allowed_extensions


def process_image(contents: bytes) -> Image.Image:
    """Load and validate image"""
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Could not open image: {str(e)}")


def validate_image_dimensions(image: Image.Image, min_size: int = 32) -> bool:
    """Validate image dimensions"""
    return image.size[0] >= min_size and image.size[1] >= min_size


# ======================== API Endpoints ========================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_version="v1.0",
        device=str(DEVICE),
        timestamp=datetime.now().isoformat()
    )


@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    return ModelInfoResponse(
        model_name="ResNet-18 ADAS Detector",
        version="1.0",
        num_classes=NUM_CLASSES,
        classes=CLASS_NAMES,
        input_shape="128x128x3",
        device=str(DEVICE)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict class of uploaded image.

    Args:
        file: Image file (JPEG, PNG, GIF, BMP)

    Returns:
        PredictionResponse with prediction, confidence, and class probabilities
    """
    start_time = time.time()

    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()

        if not validate_image_file(file_ext):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file_ext}. Allowed: jpg, jpeg, png, gif, bmp"
            )

        # Read image
        try:
            contents = await file.read()
        except Exception as read_error:
            logger.error(f"Error reading file: {str(read_error)}")
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(read_error)}")

        # Check if file is empty
        if not contents or len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file - no data received")

        # Process image
        try:
            image = process_image(contents)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate dimensions
        if not validate_image_dimensions(image):
            raise HTTPException(
                status_code=400,
                detail=f"Image too small: {image.size}. Minimum: 32x32"
            )

        # Transform and predict
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_value = confidence.item()

        # Build class probabilities dict
        class_probs = {
            class_name: float(probabilities[0, idx].item())
            for idx, class_name in enumerate(CLASS_NAMES)
        }

        latency = (time.time() - start_time) * 1000

        logger.info(f"Prediction: {predicted_class} (confidence: {confidence_value:.4f})")

        return PredictionResponse(
            prediction=predicted_class,
            confidence=round(confidence_value, 4),
            class_probabilities=class_probs,
            model_version="v1.0",
            latency_ms=round(latency, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================== Main ========================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting API server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
