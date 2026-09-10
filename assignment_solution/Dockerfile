# Builder contains pip tooling; the runtime image contains only application dependencies.
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --target /deps -r requirements.txt

FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /deps /usr/local/lib/python3.11/site-packages
COPY app.py ./
COPY models/ ./models/
USER appuser
EXPOSE 8080
ENV MODEL_PATH=/app/models/model.int8.onnx PYTHONUNBUFFERED=1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
