"""Create a dynamic INT8 ONNX model and enforce a held-out accuracy gate."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

def predict(path: Path, features: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.ravel(session.run(None, {session.get_inputs()[0].name: features.astype(np.float32)})[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-model", type=Path, required=True)
    parser.add_argument("--int8-model", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True, help="NPZ with features and labels")
    parser.add_argument("--max-accuracy-drop", type=float, default=.01)
    args = parser.parse_args()
    quantize_dynamic(str(args.fp32_model), str(args.int8_model), weight_type=QuantType.QInt8)
    data = np.load(args.validation_data)
    labels = data["labels"]
    fp32_accuracy = float(((predict(args.fp32_model, data["features"]) >= .5) == labels).mean())
    int8_accuracy = float(((predict(args.int8_model, data["features"]) >= .5) == labels).mean())
    report = {"fp32_accuracy": fp32_accuracy, "int8_accuracy": int8_accuracy,
              "accuracy_drop": fp32_accuracy - int8_accuracy,
              "fp32_size_mb": args.fp32_model.stat().st_size / 1e6,
              "int8_size_mb": args.int8_model.stat().st_size / 1e6}
    print(json.dumps(report, indent=2))
    if report["accuracy_drop"] > args.max_accuracy_drop:
        raise SystemExit("INT8 accuracy drop exceeds the release threshold")
