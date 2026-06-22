from __future__ import annotations

from typing import Any

from graph_benchmark.clustering.kmeans import FeatureKMeansConfig, cluster_node_features
from graph_benchmark.models.arga import ARGATrainingConfig, train_arga_embeddings
from graph_benchmark.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("arga_kmeans_embeddings")
def run_arga_kmeans_embeddings(
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

    n_clusters_config = params.get("n_clusters", "auto")
    n_clusters = int(dataset.num_classes) if n_clusters_config == "auto" else int(n_clusters_config)
    clustering = cluster_node_features(
        arga_output["embeddings"],
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
            "name": "arga_kmeans_embeddings",
            "input": "arga_latent_embeddings",
            "uses_topology": True,
            "n_clusters": n_clusters,
            "random_state": int(params.get("random_state", 42)),
            "n_init": int(params.get("n_init", 20)),
            "max_iter": int(params.get("max_iter", 300)),
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
            "device": arga_output["device"],
        },
        "extra_metrics": {
            "reconstruction_loss": arga_output["final_reconstruction_loss"],
            "clustering_loss": arga_output["final_regularization_loss"],
            "total_loss": arga_output["final_total_loss"],
            "arga_discriminator_loss": arga_output["final_discriminator_loss"],
        },
    }
