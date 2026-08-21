"""Train a classification pipeline and save run artifacts for reproducibility."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split

from src.eval.metrics import classification_metrics
from src.models.trainer import train_model
from src.utils.common import ensure_dir, load_yaml, set_seed
from src.utils.mlflow_utils import (
    configure_mlflow,
    log_artifact_if_exists,
    log_artifacts_dir_if_exists,
    log_dict_metrics,
    log_dict_params,
    log_sklearn_model,
    maybe_start_run,
    resolve_registered_model_name,
)


def _bootstrap_dataframe(name: str) -> pd.DataFrame:
    """Load a binary-classification DataFrame from a named sklearn dataset."""
    if name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        return data.frame.copy()

    if name == "wine":
        # wine has 3 classes; binarise: class 0 → 1, classes 1/2 → 0.
        data = load_wine(as_frame=True)
        df = data.frame.copy()
        df["target"] = (df["target"] == 0).astype(int)
        return df

    if name == "iris":
        # iris has 3 classes; keep only setosa (0) vs versicolor (1).
        data = load_iris(as_frame=True)
        df = data.frame.copy()
        df = df[df["target"].isin([0, 1])].reset_index(drop=True)
        return df

    if name == "synthetic":
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            n_redundant=5, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
        return df

    raise ValueError(
        f"Unknown bootstrap_dataset '{name}'. "
        "Choose from: breast_cancer, wine, iris, synthetic."
    )


def _ensure_training_data(data_cfg: dict) -> pd.DataFrame:
    dataset_path = Path(data_cfg["dataset_path"])
    if dataset_path.exists():
        return pd.read_csv(dataset_path)

    # Bootstrap a local demo dataset so the training workflow runs out-of-the-box.
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap = data_cfg.get("bootstrap_dataset", "breast_cancer")
    df = _bootstrap_dataframe(bootstrap)
    df.rename(columns={"target": data_cfg["target_column"]}, inplace=True)
    df.to_csv(dataset_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--config", default="configs/train/default.yaml")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns")
    parser.add_argument("--mlflow-experiment", default="ml-demo-supervised")
    parser.add_argument("--mlflow-run-name", default="")
    parser.add_argument("--mlflow-register-model", action="store_true")
    parser.add_argument("--mlflow-model-name", default="")
    parser.add_argument("--mlflow-model-artifact-path", default="model")
    args = parser.parse_args()

    if args.mlflow_register_model and not args.mlflow:
        raise ValueError("--mlflow-register-model requires --mlflow")

    train_cfg = load_yaml(args.config)
    data_cfg = load_yaml(train_cfg["paths"]["data_config"])
    feature_cfg = load_yaml(train_cfg["paths"]["feature_config"])
    model_cfg = load_yaml(train_cfg["paths"]["model_config"])

    mlflow = configure_mlflow(
        enabled=bool(args.mlflow),
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
    )
    mlflow_run_name = args.mlflow_run_name or str(train_cfg.get("run_name", "train_run"))

    with maybe_start_run(mlflow, run_name=mlflow_run_name):
        set_seed(train_cfg["seed"])

        df = _ensure_training_data(data_cfg)
        target_col = data_cfg["target_column"]
        drop_cols = feature_cfg.get("drop_columns", [])

        X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(data_cfg.get("test_size", 0.2)),
            random_state=int(data_cfg.get("random_state", 42)),
            stratify=y,
        )

        cv_config = train_cfg.get("cross_validation", {})
        cv_folds = int(cv_config.get("folds", 5)) if cv_config.get("enabled") else None
        cv_scoring = cv_config.get("scoring", "roc_auc")

        model, cv_results = train_model(
            X_train=X_train,
            y_train=y_train,
            model_type=model_cfg.get("model_type", "logistic_regression"),
            model_params=model_cfg.get("params", {}),
            scale_numeric=bool(feature_cfg.get("scale_numeric", True)),
            pca_enabled=bool(feature_cfg.get("pca_enabled", False)),
            pca_n_components=feature_cfg.get("pca_n_components"),
            calibrate=bool(model_cfg.get("calibrate", False)),
            cv_folds=cv_folds,
            cv_scoring=cv_scoring,
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = classification_metrics(y_test, y_pred, y_prob)
        metrics.update(cv_results)

        run_dir = ensure_dir(Path(train_cfg["paths"]["runs_dir"]) / train_cfg["run_name"])

        model_file = train_cfg["artifacts"]["model_file"]
        metrics_file = train_cfg["artifacts"]["metrics_file"]
        params_file = train_cfg["artifacts"]["params_file"]
        predictions_file = train_cfg["artifacts"]["predictions_file"]

        joblib.dump(model, run_dir / model_file)
        (run_dir / metrics_file).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        params = {
            "train_config": args.config,
            "data_config": train_cfg["paths"]["data_config"],
            "feature_config": train_cfg["paths"]["feature_config"],
            "model_config": train_cfg["paths"]["model_config"],
        }
        (run_dir / params_file).write_text(json.dumps(params, indent=2), encoding="utf-8")

    # Inference bundle descriptor – documents exactly what artifact was saved
    # and what features it expects, so serving code can validate inputs.
        bundle_info = {
            "model_type": model_cfg.get("model_type", "unknown"),
            "run_name": train_cfg["run_name"],
            "model_file": model_file,
            "features": X_train.columns.tolist(),
            "calibrated": bool(model_cfg.get("calibrate", False)),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_config": args.config,
        }
        (run_dir / "bundle_info.json").write_text(json.dumps(bundle_info, indent=2), encoding="utf-8")

        pred_df = X_test.copy()
        pred_df["y_true"] = y_test.values
        pred_df["y_pred"] = y_pred
        # Persist probabilities when available for thresholding and calibration checks.
        if y_prob is not None:
            pred_df["y_score"] = y_prob
        pred_df.to_csv(run_dir / predictions_file, index=False)

        holdout = X_test.copy()
        holdout[target_col] = y_test.values
        holdout.to_csv(run_dir / "holdout.csv", index=False)

        summary_path = Path(train_cfg["paths"]["runs_dir"]) / "summary.csv"
        cv_mean_key = f"cv_{cv_scoring}_mean" if cv_config.get("enabled") else ""
        summary_row = {
            "run_name": train_cfg["run_name"],
            "model": model_cfg.get("model_type", "unknown"),
            "calibrated": bool(model_cfg.get("calibrate", False)),
            "accuracy": metrics.get("accuracy", ""),
            "f1": metrics.get("f1", ""),
            "roc_auc": metrics.get("roc_auc", ""),
            "cv_score_mean": metrics.get(cv_mean_key, ""),
            "cv_scoring": cv_config.get("scoring", "") if cv_config.get("enabled") else "",
            "notes": "",
        }
        # Append instead of overwrite to keep a lightweight experiment ledger.
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
        else:
            summary_df = pd.DataFrame([summary_row])
        summary_df.to_csv(summary_path, index=False)

        if mlflow is not None:
            mlflow.set_tag("task", "supervised_classification")
            mlflow.set_tag("run_name", str(train_cfg.get("run_name", "")))
            mlflow.set_tag("model_type", str(model_cfg.get("model_type", "unknown")))

            log_dict_params(
                mlflow,
                {
                    "seed": train_cfg.get("seed"),
                    "train_config": args.config,
                    "data_config": train_cfg["paths"]["data_config"],
                    "feature_config": train_cfg["paths"]["feature_config"],
                    "model_config": train_cfg["paths"]["model_config"],
                    "test_size": data_cfg.get("test_size"),
                    "split_random_state": data_cfg.get("random_state"),
                    "model_type": model_cfg.get("model_type", "logistic_regression"),
                    "calibrate": bool(model_cfg.get("calibrate", False)),
                    "cv_enabled": bool(cv_config.get("enabled", False)),
                    "cv_folds": cv_folds,
                    "cv_scoring": cv_scoring,
                },
            )
            log_dict_params(mlflow, model_cfg.get("params", {}), prefix="model_param_")
            log_dict_metrics(mlflow, metrics)

            log_artifact_if_exists(mlflow, args.config, artifact_path="configs")
            log_artifact_if_exists(mlflow, train_cfg["paths"]["data_config"], artifact_path="configs")
            log_artifact_if_exists(mlflow, train_cfg["paths"]["feature_config"], artifact_path="configs")
            log_artifact_if_exists(mlflow, train_cfg["paths"]["model_config"], artifact_path="configs")
            log_artifacts_dir_if_exists(mlflow, run_dir, artifact_path="run_artifacts")

            registered_model_name = resolve_registered_model_name(
                enable_registration=bool(args.mlflow_register_model),
                model_name=args.mlflow_model_name,
                fallback_name=f"{train_cfg.get('run_name', 'model')}_sklearn",
            )
            log_sklearn_model(
                mlflow_module=mlflow,
                model=model,
                artifact_path=args.mlflow_model_artifact_path,
                registered_model_name=registered_model_name,
            )

        print(f"Saved run artifacts to: {run_dir}")
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
