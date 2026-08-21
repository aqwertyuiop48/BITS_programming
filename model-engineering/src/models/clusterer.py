"""Unsupervised model builders for clustering workflows."""

from __future__ import annotations

from sklearn.cluster import DBSCAN, KMeans

_CLUSTER_MODEL_REGISTRY: dict = {
    "kmeans": KMeans,
    "dbscan": DBSCAN,
}


def build_cluster_model(model_type: str, model_params: dict):
    """Instantiate a clustering model from the registry."""
    if model_type not in _CLUSTER_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown clustering model_type '{model_type}'. "
            f"Available: {sorted(_CLUSTER_MODEL_REGISTRY)}"
        )

    cls = _CLUSTER_MODEL_REGISTRY[model_type]
    return cls(**dict(model_params))
