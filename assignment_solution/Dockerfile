# Stage 1: Builder — installs ONLY the application dependencies.
# Quantization/validation is NOT repeated here: CI already ran the quality
# gate (optimize_model.py) and produced models/model.int8.onnx before the
# image build, so no training data or modelling tooling is needed at build time.
# Pinned to an LTS Debian base (bookworm, Debian 12) rather than the floating
# slim tag (presently trixie/Debian 13): LTS gets long-lived, well-maintained
# security backports, so fewer unfixable OS CVEs surface.
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Copy dependency definition and install directly to /deps
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --target /deps -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim-bookworm AS runtime

# Hardening: pull the latest Debian security backports before anything else,
# then drop the package lists so no apt metadata ships in the image.
# This clears the OS CVEs that already have a published fix.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Remove the base image's bundled packaging stack (pip/setuptools/wheel and
# their orphaned jaraco.* deps). They are build tooling we do not need at
# runtime, and leaving them introduces stale, vulnerable versions (e.g. wheel
# 0.45.1, jaraco.context 5.3.0) that surface in the CVE scan. All runtime
# Python dependencies come from /deps below, so wiping site-packages first is
# safe and deterministic - and no package manager ships in the runtime image.
RUN rm -rf /usr/local/lib/python3.11/site-packages/*

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