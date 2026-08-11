"""Evaluation metric utilities for binary classification tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        unique = np.unique(y_true)
        # ROC-AUC is only defined for binary targets in this helper.
        if len(unique) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def clustering_metrics(X, labels) -> dict:
    """Compute robust clustering metrics with safeguards for edge cases."""
    labels_array = np.asarray(labels)
    out = {
        "n_samples": int(labels_array.shape[0]),
        "noise_count": int(np.sum(labels_array == -1)),
    }
    out["noise_ratio"] = (
        float(out["noise_count"] / out["n_samples"])
        if out["n_samples"]
        else 0.0
    )

    unique_non_noise = np.unique(labels_array[labels_array != -1])
    out["cluster_count"] = int(len(unique_non_noise))

    # Many clustering metrics are undefined when fewer than two clusters exist.
    mask = labels_array != -1
    if out["noise_count"] == 0:
        mask = np.ones_like(labels_array, dtype=bool)

    labels_eval = labels_array[mask]
    x_eval = X[mask]
    unique_eval = np.unique(labels_eval)
    if len(unique_eval) < 2:
        out["silhouette"] = None
        out["davies_bouldin"] = None
        out["calinski_harabasz"] = None
        out["metric_note"] = "Metrics undefined: fewer than 2 clusters after excluding noise."
        return out

    out["silhouette"] = float(silhouette_score(x_eval, labels_eval))
    out["davies_bouldin"] = float(davies_bouldin_score(x_eval, labels_eval))
    out["calinski_harabasz"] = float(calinski_harabasz_score(x_eval, labels_eval))
    return out
