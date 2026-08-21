"""Optional MLflow helpers used by training/evaluation entrypoints."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any


def _import_mlflow():
    try:
        import mlflow  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "MLflow is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return mlflow


def configure_mlflow(enabled: bool, tracking_uri: str, experiment_name: str):
    """Return configured mlflow module when enabled, otherwise None."""
    if not enabled:
        return None

    mlflow = _import_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def maybe_start_run(mlflow_module, run_name: str):
    """Return a context manager for an MLflow run or a no-op context."""
    if mlflow_module is None:
        return nullcontext(None)
    return mlflow_module.start_run(run_name=run_name)


def log_dict_params(mlflow_module, values: dict[str, Any], prefix: str = "") -> None:
    """Log small dictionaries as MLflow params (stringified where necessary)."""
    if mlflow_module is None:
        return

    for key, value in values.items():
        param_key = f"{prefix}{key}" if prefix else key
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            mlflow_module.log_param(param_key, value)
        else:
            mlflow_module.log_param(param_key, str(value))


def log_dict_metrics(mlflow_module, metrics: dict[str, Any], prefix: str = "") -> None:
    """Log numeric dictionary values as MLflow metrics."""
    if mlflow_module is None:
        return

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            metric_key = f"{prefix}{key}" if prefix else key
            mlflow_module.log_metric(metric_key, float(value))


def log_artifact_if_exists(mlflow_module, path: str | Path, artifact_path: str | None = None) -> None:
    """Log a single artifact file if it exists."""
    if mlflow_module is None:
        return

    p = Path(path)
    if p.exists() and p.is_file():
        mlflow_module.log_artifact(str(p), artifact_path=artifact_path)


def log_artifacts_dir_if_exists(mlflow_module, path: str | Path, artifact_path: str | None = None) -> None:
    """Log an artifact directory if it exists."""
    if mlflow_module is None:
        return

    p = Path(path)
    if p.exists() and p.is_dir():
        mlflow_module.log_artifacts(str(p), artifact_path=artifact_path)


def resolve_registered_model_name(enable_registration: bool, model_name: str, fallback_name: str) -> str | None:
    """Resolve the registry model name when registration is enabled."""
    if not enable_registration:
        return None

    cleaned = model_name.strip()
    if cleaned:
        return cleaned
    return fallback_name


def log_sklearn_model(
    mlflow_module,
    model,
    artifact_path: str,
    registered_model_name: str | None = None,
) -> None:
    """Log sklearn model and optionally register it; never fail training on registry errors."""
    if mlflow_module is None:
        return

    try:
        mlflow_module.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
    except Exception as exc:  # pragma: no cover - backend-specific behavior
        print(f"MLflow model logging/registration skipped: {exc}")


def log_pytorch_model(
    mlflow_module,
    model,
    artifact_path: str,
    registered_model_name: str | None = None,
) -> None:
    """Log pytorch model and optionally register it; never fail training on registry errors."""
    if mlflow_module is None:
        return

    try:
        mlflow_module.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
    except Exception as exc:  # pragma: no cover - backend-specific behavior
        print(f"MLflow model logging/registration skipped: {exc}")
