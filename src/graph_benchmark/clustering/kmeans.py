from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from graph_benchmark.registry import ALGORITHM_REGISTRY


@dataclass(frozen=True)
class FeatureKMeansConfig:
    n_clusters: int
    random_state: int = 42
    n_init: int = 20
    max_iter: int = 300
    scale_features: bool = True


def cluster_node_features(
    features: np.ndarray,
    config: FeatureKMeansConfig,
) -> dict[str, Any]:
    processed_features = features

    if config.scale_features:
        scaler = StandardScaler()
        processed_features = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=config.n_clusters,
        n_init=config.n_init,
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    assignments = model.fit_predict(processed_features)

    return {
        "assignments": assignments,
        "features": processed_features,
        "inertia": float(model.inertia_),
    }


@ALGORITHM_REGISTRY.register("kmeans_features")
def run_kmeans_features(
    dataset_bundle: dict[str, Any],
    algorithm_config: dict[str, Any],
) -> dict[str, Any]:
    dataset = dataset_bundle["dataset"]
    data = dataset_bundle["data"]
    params = algorithm_config.get("params", {})
    features = data.x.detach().cpu().numpy()

    n_clusters_config = params.get("n_clusters", "auto")
    n_clusters = int(dataset.num_classes) if n_clusters_config == "auto" else int(n_clusters_config)

    clustering = cluster_node_features(
        features,
        FeatureKMeansConfig(
            n_clusters=n_clusters,
            random_state=int(params.get("random_state", 42)),
            n_init=int(params.get("n_init", 20)),
            max_iter=int(params.get("max_iter", 300)),
            scale_features=bool(params.get("scale_features", True)),
        ),
    )

    return {
        "assignments": clustering["assignments"],
        "features": clustering["features"],
        "inertia": clustering["inertia"],
        "metadata": {
            "name": "kmeans_features",
            "input": "node_features_only",
            "ignores_topology": True,
            "n_clusters": n_clusters,
            "random_state": int(params.get("random_state", 42)),
            "n_init": int(params.get("n_init", 20)),
            "max_iter": int(params.get("max_iter", 300)),
            "scale_features": bool(params.get("scale_features", True)),
        },
    }
