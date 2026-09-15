# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy dependency definition and install directly to /deps
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --target /deps -r requirements.txt

# Copy source code and model optimization script
COPY optimize_model.py .
COPY models/ ./models/
COPY data/ ./data/

# Run quantization inside builder stage so model.int8.onnx is guaranteed to exist
RUN PYTHONPATH=/deps python optimize_model.py \
    --fp32-model models/model.onnx \
    --int8-model models/model.int8.onnx \
    --validation-data data/validation.npz \
    --max-accuracy-drop 0.01

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Pre-create app directory and unprivileged user
RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /deps /usr/local/lib/python3.11/site-packages

# Copy application code and optimized model artifacts
COPY app.py ./
COPY --from=builder /build/models/ /app/models/

# Ensure runtime user owns the copied application files
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

ENV MODEL_PATH=/app/models/model.int8.onnx \
    PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]