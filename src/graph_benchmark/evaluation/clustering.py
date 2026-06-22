from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix


def compute_purity(labels: np.ndarray, assignments: np.ndarray) -> float:
    contingency = contingency_matrix(labels, assignments)
    return float(np.sum(np.max(contingency, axis=0)) / np.sum(contingency))


def evaluate_clustering(
    features: np.ndarray | None,
    labels: np.ndarray,
    assignments: np.ndarray,
) -> dict[str, float | dict[str, int]]:
    unique_assignments = np.unique(assignments)
    cluster_distribution = {
        str(cluster_id): count
        for cluster_id, count in sorted(Counter(assignments).items())
    }

    metrics: dict[str, float | dict[str, int]] = {
        "nmi": float(normalized_mutual_info_score(labels, assignments)),
        "ari": float(adjusted_rand_score(labels, assignments)),
        "purity": compute_purity(labels, assignments),
        "num_clusters_found": int(len(unique_assignments)),
        "cluster_distribution": cluster_distribution,
    }

    if features is not None and 1 < len(unique_assignments) < len(assignments):
        metrics["silhouette"] = float(silhouette_score(features, assignments))

    return metrics
