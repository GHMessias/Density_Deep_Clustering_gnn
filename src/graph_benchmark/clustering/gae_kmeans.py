from __future__ import annotations

from typing import Any

from graph_benchmark.clustering.kmeans import FeatureKMeansConfig, cluster_node_features
from graph_benchmark.models.gae import GAETrainingConfig, train_gae_embeddings
from graph_benchmark.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("gae_kmeans_embeddings")
def run_gae_kmeans_embeddings(
    dataset_bundle: dict[str, Any],
    algorithm_config: dict[str, Any],
) -> dict[str, Any]:
    dataset = dataset_bundle["dataset"]
    data = dataset_bundle["data"]
    params = algorithm_config.get("params", {})

    gae_output = train_gae_embeddings(
        data,
        GAETrainingConfig(
            input_channels=int(dataset.num_features),
            hidden_channels=int(params.get("encoder_hidden_channels", 128)),
            embedding_channels=int(params.get("embedding_channels", 32)),
            epochs=int(params.get("epochs", 200)),
            learning_rate=float(params.get("learning_rate", 0.01)),
            weight_decay=float(params.get("weight_decay", 0.0)),
            dropout=float(params.get("dropout", 0.0)),
            device=str(params.get("device", "auto")),
        ),
    )

    n_clusters_config = params.get("n_clusters", "auto")
    n_clusters = int(dataset.num_classes) if n_clusters_config == "auto" else int(n_clusters_config)
    clustering = cluster_node_features(
        gae_output["embeddings"],
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
            "name": "gae_kmeans_embeddings",
            "input": "gae_latent_embeddings",
            "uses_topology": True,
            "n_clusters": n_clusters,
            "random_state": int(params.get("random_state", 42)),
            "n_init": int(params.get("n_init", 20)),
            "max_iter": int(params.get("max_iter", 300)),
            "scale_embeddings": bool(params.get("scale_embeddings", True)),
            "encoder_hidden_channels": int(params.get("encoder_hidden_channels", 128)),
            "embedding_channels": int(params.get("embedding_channels", 32)),
            "epochs": int(params.get("epochs", 200)),
            "learning_rate": float(params.get("learning_rate", 0.01)),
            "weight_decay": float(params.get("weight_decay", 0.0)),
            "dropout": float(params.get("dropout", 0.0)),
            "device": gae_output["device"],
        },
        "extra_metrics": {
            "reconstruction_loss": gae_output["final_reconstruction_loss"],
        },
    }
