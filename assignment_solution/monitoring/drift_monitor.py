"""KS-test drift detection for continuous recommendation-model features."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp

def detect_drift(training: np.ndarray, production: np.ndarray, alpha: float = .05) -> dict:
    if training.ndim != 2 or production.ndim != 2 or training.shape[1] != production.shape[1]:
        raise ValueError("arrays must be 2-D and have matching feature counts")
    features = []
    for index in range(training.shape[1]):
        statistic, p_value = ks_2samp(training[:, index], production[:, index])
        features.append({"feature_index": index, "ks_statistic": float(statistic),
                         "p_value": float(p_value), "drifted": bool(p_value < alpha)})
    return {"alpha": alpha, "drifted_features": sum(x["drifted"] for x in features), "features": features}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-baseline", type=Path, required=True)
    parser.add_argument("--production-window", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args()
    print(json.dumps(detect_drift(np.load(args.training_baseline), np.load(args.production_window), args.alpha), indent=2))
