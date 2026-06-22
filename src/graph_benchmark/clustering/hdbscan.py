from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class EmbeddingHDBSCANConfig:
    min_cluster_size: int = 20
    min_samples: int = 10
    cluster_selection_method: str = "eom"
    scale_features: bool = True


def cluster_embeddings_hdbscan(
    features: np.ndarray,
    config: EmbeddingHDBSCANConfig,
) -> dict[str, Any]:
    processed_features = features

    if config.scale_features:
        scaler = StandardScaler()
        processed_features = scaler.fit_transform(features)

    model = HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_method=config.cluster_selection_method,
        copy=True,
    )
    assignments = model.fit_predict(processed_features)
    unique_clusters = {int(label) for label in np.unique(assignments) if int(label) >= 0}

    return {
        "assignments": assignments,
        "features": processed_features,
        "num_clusters_found": int(len(unique_clusters)),
        "noise_ratio": float(np.mean(assignments == -1)),
    }
