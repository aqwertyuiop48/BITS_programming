"""FastAPI inference service for the optimized recommendation classifier."""
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.int8.onnx")
session: ort.InferenceSession | None = None


class RecommendationRequest(BaseModel):
    features: list[float] = Field(min_length=1, description="Ordered model feature vector")


class RecommendationResponse(BaseModel):
    prediction: int
    score: float
    model_version: str
    latency_ms: float


@asynccontextmanager
async def lifespan(_: FastAPI):
    global session
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    yield
    session = None


app = FastAPI(title="Recommendation Inference API", version=os.getenv("MODEL_VERSION", "candidate"), lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    if session is None:
        raise HTTPException(status_code=503, detail="model not ready")
    return {"status": "ok"}


@app.post("/predict", response_model=RecommendationResponse)
def predict(request: RecommendationRequest) -> RecommendationResponse:
    if session is None:
        raise HTTPException(status_code=503, detail="model not ready")
    started = time.perf_counter()
    input_meta = session.get_inputs()[0]
    features = np.asarray(request.features, dtype=np.float32).reshape(1, -1)
    expected_features = input_meta.shape[-1]
    if isinstance(expected_features, int) and features.shape[1] != expected_features:
        raise HTTPException(status_code=422, detail=f"expected {expected_features} features")
    output = session.run(None, {input_meta.name: features})[0]
    score = float(np.ravel(output)[0])
    return RecommendationResponse(
        prediction=int(score >= 0.5), score=score,
        model_version=app.version, latency_ms=(time.perf_counter() - started) * 1000,
    )
