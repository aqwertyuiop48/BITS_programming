"""Model training routines that build and fit sklearn pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.features.preprocess import build_preprocessor

_MODEL_REGISTRY: dict = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "svm": SVC,
}


def build_model(model_type: str, model_params: dict, calibrate: bool = False):
    """Instantiate a classifier from its registry name and params dict.

    SVM gets ``probability=True`` injected when *not* wrapping with
    CalibratedClassifierCV so that ``predict_proba`` is always available.
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {sorted(_MODEL_REGISTRY)}"
        )
    cls = _MODEL_REGISTRY[model_type]
    params = dict(model_params)
    # SVM does not expose predict_proba by default; inject only when not using
    # CalibratedClassifierCV (which would raise a double-calibration warning).
    if model_type == "svm" and not calibrate:
        params.setdefault("probability", True)
    estimator = cls(**params)
    if calibrate:
        estimator = CalibratedClassifierCV(estimator, cv=5, method="isotonic")
    return estimator


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    model_params: dict,
    scale_numeric: bool = True,
    pca_enabled: bool = False,
    pca_n_components: int | None = None,
    calibrate: bool = False,
    cv_folds: int | None = None,
    cv_scoring: str = "roc_auc",
) -> tuple[Pipeline, dict]:
    """Build, optionally CV-score, and fit a full sklearn pipeline.

    Parameters
    ----------
    cv_folds:
        Number of stratified CV folds.  ``None`` skips cross-validation.

    Returns
    -------
    pipeline : fitted Pipeline (preprocessor [→ PCA] → model)
    cv_results : dict of CV scores (empty when cv_folds is None / < 2)
    """
    preprocessor = build_preprocessor(X_train, scale_numeric=scale_numeric)
    estimator = build_model(model_type, model_params, calibrate=calibrate)

    steps = [("preprocessor", preprocessor)]
    if pca_enabled:
        steps.append(("pca", PCA(n_components=pca_n_components, random_state=42)))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps=steps)

    cv_results: dict = {}
    if cv_folds and cv_folds > 1:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=cv_scoring)
        cv_results = {
            f"cv_{cv_scoring}_mean": float(np.mean(scores)),
            f"cv_{cv_scoring}_std": float(np.std(scores)),
            f"cv_{cv_scoring}_folds": scores.tolist(),
        }

    pipeline.fit(X_train, y_train)
    return pipeline, cv_results


# ---------------------------------------------------------------------------
# Backward-compatible shim – keeps existing callers working unchanged
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
    scale_numeric: bool,
    pca_enabled: bool = False,
    pca_n_components: int | None = None,
) -> Pipeline:
    pipeline, _ = train_model(
        X_train=X_train,
        y_train=y_train,
        model_type="logistic_regression",
        model_params=model_params,
        scale_numeric=scale_numeric,
        pca_enabled=pca_enabled,
        pca_n_components=pca_n_components,
    )
    return pipeline
