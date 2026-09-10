# Module 05 production recommendation-model solution

## Included deliverables

- `Dockerfile` and `app.py`: secure multi-stage FastAPI/ONNX Runtime service.
- `optimize_model.py`: dynamic INT8 conversion plus release-quality gate.
- `monitoring/drift_monitor.py`: KS-test drift detector.
- `report.md` and `architecture.mmd`: the requested rationale and diagram.
- `.github/workflows/ci.yml`: CI/CD reference implementation.

## Commands

```bash
pip install -r requirements.txt scipy
python optimize_model.py --fp32-model models/model.onnx --int8-model models/model.int8.onnx --validation-data data/validation.npz
python monitoring/drift_monitor.py --training-baseline data/training_features.npy --production-window data/live_window.npy
docker build -t recommendation:local .
docker run -p 8080:8080 -v "$(pwd)/models:/app/models:ro" recommendation:local
```

The actual model and validation data should be fetched from the model registry
by CI and must not be committed to this repository.
