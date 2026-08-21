"""Evaluate a saved run on its holdout set and write evaluation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.eval.metrics import classification_metrics
from src.inference.predictor import load_model
from src.utils.mlflow_utils import (
    configure_mlflow,
    log_artifact_if_exists,
    log_dict_metrics,
    log_dict_params,
    maybe_start_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved run")
    parser.add_argument("--run-dir", default="runs/run_001")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns")
    parser.add_argument("--mlflow-experiment", default="ml-demo-evaluation")
    parser.add_argument("--mlflow-run-name", default="")
    args = parser.parse_args()

    mlflow = configure_mlflow(
        enabled=bool(args.mlflow),
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
    )
    run_name = args.mlflow_run_name or f"evaluate:{Path(args.run_dir).name}"

    with maybe_start_run(mlflow, run_name=run_name):
        run_dir = Path(args.run_dir)
        model = load_model(run_dir)

        holdout_path = run_dir / "holdout.csv"
        if not holdout_path.exists():
            raise FileNotFoundError(f"Missing holdout file: {holdout_path}")

        df = pd.read_csv(holdout_path)
        if args.target_col not in df.columns:
            raise ValueError(f"Target column '{args.target_col}' not found in holdout.csv")

        X = df.drop(columns=[args.target_col])
        y_true = df[args.target_col]

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = classification_metrics(y_true, y_pred, y_prob)

        out_path = run_dir / "evaluation_metrics.json"
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if mlflow is not None:
            mlflow.set_tag("task", "evaluation")
            mlflow.set_tag("run_dir", str(run_dir))
            log_dict_params(
                mlflow,
                {
                    "run_dir": str(run_dir),
                    "target_col": args.target_col,
                },
            )
            log_dict_metrics(mlflow, metrics, prefix="eval_")
            log_artifact_if_exists(mlflow, out_path, artifact_path="evaluation")

        print(f"Evaluation metrics saved to: {out_path}")
        print(metrics)


if __name__ == "__main__":
    main()
