# Stage 1: Builder — installs ONLY the application dependencies.
# Quantization/validation is NOT repeated here: CI already ran the quality
# gate (optimize_model.py) and produced models/model.int8.onnx before the
# image build, so no training data or modelling tooling is needed at build time.
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy dependency definition and install directly to /deps
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --target /deps -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Pre-create app directory and unprivileged user
RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /deps /usr/local/lib/python3.11/site-packages

# Copy application code and the registry-approved optimized model artifacts
# (models/ carries both model.onnx and model.int8.onnx produced by CI)
COPY app.py ./
COPY models/ ./models/

# Ensure runtime user owns the copied application files
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

ENV MODEL_PATH=/app/models/model.int8.onnx \
    PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]