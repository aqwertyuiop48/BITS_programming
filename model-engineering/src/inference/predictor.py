"""Inference helpers for loading trained models and producing prediction tables."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def load_model(run_dir: str | Path, model_file: str = "model.joblib"):
    model_path = Path(run_dir) / model_file
    return joblib.load(model_path)


def predict_dataframe(model, X: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(X)
    out = pd.DataFrame({"prediction": preds})
    # Include positive-class score for ranking and threshold tuning workflows.
    if hasattr(model, "predict_proba"):
        out["score"] = model.predict_proba(X)[:, 1]
    return out
