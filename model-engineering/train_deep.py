"""Train deep learning demo models (CNN/RNN/LSTM/GRU) with YAML configs.

This mirrors the artifact contract of train.py:
- model file
- metrics.json
- params.json
- predictions.csv
- bundle_info.json
- runs/summary.csv append
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.deep_trainer import load_demo_dataset, train_deep_model
from src.utils.common import ensure_dir, load_yaml, set_seed
from src.utils.mlflow_utils import (
    configure_mlflow,
    log_artifact_if_exists,
    log_artifacts_dir_if_exists,
    log_dict_metrics,
    log_dict_params,
    log_pytorch_model,
    maybe_start_run,
    resolve_registered_model_name,
)


def _to_metrics(y_true: np.ndarray, y_pred: np.ndarray, history: dict) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "epochs": int(history.get("epochs", 0)),
        "final_train_loss": float(history.get("train_loss", [0.0])[-1]),
        "best_test_accuracy": float(max(history.get("test_accuracy", [0.0]))),
        "device": history.get("device", "cpu"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deep learning demo model")
    parser.add_argument("--config", required=True, help="Path to configs/train/*.yaml")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns")
    parser.add_argument("--mlflow-experiment", default="ml-demo-deep")
    parser.add_argument("--mlflow-run-name", default="")
    parser.add_argument("--mlflow-register-model", action="store_true")
    parser.add_argument("--mlflow-model-name", default="")
    parser.add_argument("--mlflow-model-artifact-path", default="model")
    args = parser.parse_args()

    if args.mlflow_register_model and not args.mlflow:
        raise ValueError("--mlflow-register-model requires --mlflow")

    train_cfg = load_yaml(args.config)
    data_cfg = load_yaml(train_cfg["paths"]["data_config"])
    model_cfg = load_yaml(train_cfg["paths"]["model_config"])

    mlflow = configure_mlflow(
        enabled=bool(args.mlflow),
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
    )
    mlflow_run_name = args.mlflow_run_name or str(train_cfg.get("run_name", "train_deep_run"))

    with maybe_start_run(mlflow, run_name=mlflow_run_name):
        seed = int(train_cfg.get("seed", 42))
        set_seed(seed)
        torch.manual_seed(seed)

        dataset = load_demo_dataset(data_cfg=data_cfg, seed=seed)
        model_type = model_cfg.get("model_type", "cnn2d")
        model_params = model_cfg.get("params", {})

        model, history, y_pred, y_prob = train_deep_model(
            dataset=dataset,
            model_type=model_type,
            model_params=model_params,
            train_params=train_cfg.get("training", {}),
            seed=seed,
        )

        metrics = _to_metrics(dataset.y_test, y_pred, history)

        run_dir = ensure_dir(Path(train_cfg["paths"]["runs_dir"]) / train_cfg["run_name"])

        model_file = train_cfg["artifacts"]["model_file"]
        metrics_file = train_cfg["artifacts"]["metrics_file"]
        params_file = train_cfg["artifacts"]["params_file"]
        predictions_file = train_cfg["artifacts"]["predictions_file"]

        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_type": model_type,
                "model_params": model_params,
                "classes": dataset.classes,
                "framing": dataset.framing,
                "input_shape": dataset.input_shape,
            },
            run_dir / model_file,
        )

        (run_dir / metrics_file).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        params = {
            "train_config": args.config,
            "data_config": train_cfg["paths"]["data_config"],
            "model_config": train_cfg["paths"]["model_config"],
            "seed": seed,
        }
        (run_dir / params_file).write_text(json.dumps(params, indent=2), encoding="utf-8")

        pred_df = pd.DataFrame({
            "y_true": dataset.y_test,
            "y_pred": y_pred,
        })
        for idx, cls in enumerate(dataset.classes):
            pred_df[f"p_{cls}"] = y_prob[:, idx]
        pred_df.to_csv(run_dir / predictions_file, index=False)

        history_df = pd.DataFrame(
            {
                "epoch": range(1, len(history["train_loss"]) + 1),
                "train_loss": history["train_loss"],
                "test_accuracy": history["test_accuracy"],
            }
        )
        history_df.to_csv(run_dir / "training_curve.csv", index=False)

        bundle_info = {
            "model_type": model_type,
            "run_name": train_cfg["run_name"],
            "model_file": model_file,
            "dataset_name": data_cfg.get("dataset_name", "digits"),
            "framing": dataset.framing,
            "input_shape": dataset.input_shape,
            "num_classes": len(dataset.classes),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_config": args.config,
        }
        (run_dir / "bundle_info.json").write_text(json.dumps(bundle_info, indent=2), encoding="utf-8")

        summary_path = Path(train_cfg["paths"]["runs_dir"]) / "summary.csv"
        summary_row = {
            "run_name": train_cfg["run_name"],
            "model": model_type,
            "calibrated": False,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1_macro"],
            "roc_auc": "",
            "cv_score_mean": "",
            "cv_scoring": "",
            "notes": f"deep:{data_cfg.get('dataset_name', 'digits')}:{dataset.framing}",
        }
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
        else:
            summary_df = pd.DataFrame([summary_row])
        summary_df.to_csv(summary_path, index=False)

        if mlflow is not None:
            mlflow.set_tag("task", "deep_learning")
            mlflow.set_tag("run_name", str(train_cfg.get("run_name", "")))
            mlflow.set_tag("model_type", model_type)
            log_dict_params(
                mlflow,
                {
                    "seed": seed,
                    "train_config": args.config,
                    "data_config": train_cfg["paths"]["data_config"],
                    "model_config": train_cfg["paths"]["model_config"],
                    "dataset_name": data_cfg.get("dataset_name", "digits"),
                },
            )
            log_dict_params(mlflow, model_params, prefix="model_param_")
            log_dict_params(mlflow, train_cfg.get("training", {}), prefix="train_param_")
            log_dict_metrics(mlflow, metrics)

            log_artifact_if_exists(mlflow, args.config, artifact_path="configs")
            log_artifact_if_exists(mlflow, train_cfg["paths"]["data_config"], artifact_path="configs")
            log_artifact_if_exists(mlflow, train_cfg["paths"]["model_config"], artifact_path="configs")
            log_artifacts_dir_if_exists(mlflow, run_dir, artifact_path="run_artifacts")

            registered_model_name = resolve_registered_model_name(
                enable_registration=bool(args.mlflow_register_model),
                model_name=args.mlflow_model_name,
                fallback_name=f"{train_cfg.get('run_name', 'model')}_pytorch",
            )
            log_pytorch_model(
                mlflow_module=mlflow,
                model=model,
                artifact_path=args.mlflow_model_artifact_path,
                registered_model_name=registered_model_name,
            )

        print(f"Saved run artifacts to: {run_dir}")
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
