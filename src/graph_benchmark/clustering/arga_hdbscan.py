from __future__ import annotations

from typing import Any

from graph_benchmark.clustering.hdbscan import EmbeddingHDBSCANConfig, cluster_embeddings_hdbscan
from graph_benchmark.models.arga import ARGATrainingConfig, train_arga_embeddings
from graph_benchmark.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("arga_hdbscan_embeddings")
def run_arga_hdbscan_embeddings(
    dataset_bundle: dict[str, Any],
    algorithm_config: dict[str, Any],
) -> dict[str, Any]:
    dataset = dataset_bundle["dataset"]
    data = dataset_bundle["data"]
    params = algorithm_config.get("params", {})

    arga_output = train_arga_embeddings(
        data,
        ARGATrainingConfig(
            input_channels=int(dataset.num_features),
            hidden_channels=int(params.get("encoder_hidden_channels", 128)),
            embedding_channels=int(params.get("embedding_channels", 32)),
            discriminator_hidden_channels=int(params.get("discriminator_hidden_channels", 64)),
            epochs=int(params.get("epochs", 200)),
            learning_rate=float(params.get("learning_rate", 0.01)),
            discriminator_learning_rate=float(params.get("discriminator_learning_rate", 0.001)),
            weight_decay=float(params.get("weight_decay", 0.0)),
            dropout=float(params.get("dropout", 0.0)),
            reg_loss_weight=float(params.get("reg_loss_weight", 1.0)),
            discriminator_steps=int(params.get("discriminator_steps", 1)),
            device=str(params.get("device", "auto")),
        ),
    )

    clustering = cluster_embeddings_hdbscan(
        arga_output["embeddings"],
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
            "name": "arga_hdbscan_embeddings",
            "input": "arga_latent_embeddings",
            "uses_topology": True,
            "scale_embeddings": bool(params.get("scale_embeddings", True)),
            "encoder_hidden_channels": int(params.get("encoder_hidden_channels", 128)),
            "embedding_channels": int(params.get("embedding_channels", 32)),
            "discriminator_hidden_channels": int(params.get("discriminator_hidden_channels", 64)),
            "epochs": int(params.get("epochs", 200)),
            "learning_rate": float(params.get("learning_rate", 0.01)),
            "discriminator_learning_rate": float(params.get("discriminator_learning_rate", 0.001)),
            "weight_decay": float(params.get("weight_decay", 0.0)),
            "dropout": float(params.get("dropout", 0.0)),
            "reg_loss_weight": float(params.get("reg_loss_weight", 1.0)),
            "discriminator_steps": int(params.get("discriminator_steps", 1)),
            "min_cluster_size": int(params.get("min_cluster_size", 20)),
            "min_samples": int(params.get("min_samples", 10)),
            "cluster_selection_method": str(params.get("cluster_selection_method", "eom")),
            "device": arga_output["device"],
        },
        "extra_metrics": {
            "reconstruction_loss": arga_output["final_reconstruction_loss"],
            "clustering_loss": arga_output["final_regularization_loss"],
            "total_loss": arga_output["final_total_loss"],
            "arga_discriminator_loss": arga_output["final_discriminator_loss"],
            "num_clusters_found": clustering["num_clusters_found"],
            "hdbscan_num_clusters": float(clustering["num_clusters_found"]),
            "hdbscan_noise_ratio": float(clustering["noise_ratio"]),
        },
    }
