from __future__ import annotations

from typing import Any

from graph_benchmark.clustering.kmeans import FeatureKMeansConfig, cluster_node_features
from graph_benchmark.models.node2vec import Node2VecTrainingConfig, train_node2vec_embeddings
from graph_benchmark.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("node2vec_kmeans_embeddings")
def run_node2vec_kmeans_embeddings(
    dataset_bundle: dict[str, Any],
    algorithm_config: dict[str, Any],
) -> dict[str, Any]:
    dataset = dataset_bundle["dataset"]
    data = dataset_bundle["data"]
    params = algorithm_config.get("params", {})

    node2vec_output = train_node2vec_embeddings(
        data,
        Node2VecTrainingConfig(
            embedding_channels=int(params.get("embedding_channels", 128)),
            walk_length=int(params.get("walk_length", 20)),
            context_size=int(params.get("context_size", 10)),
            walks_per_node=int(params.get("walks_per_node", 10)),
            num_negative_samples=int(params.get("num_negative_samples", 1)),
            p=float(params.get("p", 1.0)),
            q=float(params.get("q", 1.0)),
            sparse=bool(params.get("sparse", True)),
            epochs=int(params.get("epochs", 100)),
            batch_size=int(params.get("batch_size", 128)),
            learning_rate=float(params.get("learning_rate", 0.01)),
            num_workers=int(params.get("num_workers", 0)),
            device=str(params.get("device", "auto")),
        ),
    )

    n_clusters_config = params.get("n_clusters", "auto")
    n_clusters = int(dataset.num_classes) if n_clusters_config == "auto" else int(n_clusters_config)
    clustering = cluster_node_features(
        node2vec_output["embeddings"],
        FeatureKMeansConfig(
            n_clusters=n_clusters,
            random_state=int(params.get("random_state", 42)),
            n_init=int(params.get("n_init", 20)),
            max_iter=int(params.get("max_iter", 300)),
            scale_features=bool(params.get("scale_embeddings", True)),
        ),
    )

    return {
        "assignments": clustering["assignments"],
        "features": clustering["features"],
        "inertia": clustering["inertia"],
        "metadata": {
            "name": "node2vec_kmeans_embeddings",
            "input": "node2vec_embeddings",
            "uses_topology": True,
            "uses_attributes": False,
            "n_clusters": n_clusters,
            "random_state": int(params.get("random_state", 42)),
            "n_init": int(params.get("n_init", 20)),
            "max_iter": int(params.get("max_iter", 300)),
            "scale_embeddings": bool(params.get("scale_embeddings", True)),
            "embedding_channels": int(params.get("embedding_channels", 128)),
            "walk_length": int(params.get("walk_length", 20)),
            "context_size": int(params.get("context_size", 10)),
            "walks_per_node": int(params.get("walks_per_node", 10)),
            "num_negative_samples": int(params.get("num_negative_samples", 1)),
            "p": float(params.get("p", 1.0)),
            "q": float(params.get("q", 1.0)),
            "sparse": bool(params.get("sparse", True)),
            "epochs": int(params.get("epochs", 100)),
            "batch_size": int(params.get("batch_size", 128)),
            "learning_rate": float(params.get("learning_rate", 0.01)),
            "num_workers": int(params.get("num_workers", 0)),
            "device": node2vec_output["device"],
        },
        "extra_metrics": {
            "node2vec_loss": node2vec_output["final_loss"],
        },
    }
