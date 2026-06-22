from __future__ import annotations

from typing import Any

from graph_benchmark.clustering.hdbscan import EmbeddingHDBSCANConfig, cluster_embeddings_hdbscan
from graph_benchmark.models.gae import GAETrainingConfig, train_gae_embeddings
from graph_benchmark.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("gae_hdbscan_embeddings")
def run_gae_hdbscan_embeddings(
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

    clustering = cluster_embeddings_hdbscan(
        gae_output["embeddings"],
        EmbeddingHDBSCANConfig(
            min_cluster_size=int(params.get("min_cluster_size", 20)),
            min_samples=int(params.get("min_samples", 10)),
            cluster_selection_method=str(params.get("cluster_selection_method", "eom")),
            scale_features=bool(params.get("scale_embeddings", True)),
        ),
    )

    return {
        "assignments": clustering["assignments"],
        "features": clustering["features"],
        "inertia": None,
        "metadata": {
            "name": "gae_hdbscan_embeddings",
            "input": "gae_latent_embeddings",
            "uses_topology": True,
            "scale_embeddings": bool(params.get("scale_embeddings", True)),
            "encoder_hidden_channels": int(params.get("encoder_hidden_channels", 128)),
            "embedding_channels": int(params.get("embedding_channels", 32)),
            "epochs": int(params.get("epochs", 200)),
            "learning_rate": float(params.get("learning_rate", 0.01)),
            "weight_decay": float(params.get("weight_decay", 0.0)),
            "dropout": float(params.get("dropout", 0.0)),
            "min_cluster_size": int(params.get("min_cluster_size", 20)),
            "min_samples": int(params.get("min_samples", 10)),
            "cluster_selection_method": str(params.get("cluster_selection_method", "eom")),
            "device": gae_output["device"],
        },
        "extra_metrics": {
            "reconstruction_loss": gae_output["final_reconstruction_loss"],
            "num_clusters_found": clustering["num_clusters_found"],
            "hdbscan_num_clusters": float(clustering["num_clusters_found"]),
            "hdbscan_noise_ratio": float(clustering["noise_ratio"]),
        },
    }
