from __future__ import annotations

from typing import Any

from graph_benchmark.models.densgnn import DensGNNConfig
from graph_benchmark.models.densgnn2 import train_densgnn2
from graph_benchmark.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("densgnn2")
def run_densgnn2(
    dataset_bundle: dict[str, Any],
    algorithm_config: dict[str, Any],
) -> dict[str, Any]:
    dataset = dataset_bundle["dataset"]
    data = dataset_bundle["data"]
    params = algorithm_config.get("params", {})
    labels = data.y.detach().cpu().numpy() if getattr(data, "y", None) is not None else None

    output = train_densgnn2(
        data=data,
        labels=labels,
        config=DensGNNConfig(
            input_channels=int(dataset.num_features),
            hidden_channels=int(params.get("encoder_hidden_channels", 128)),
            embedding_channels=int(params.get("embedding_channels", 32)),
            epochs=int(params.get("epochs", 100)),
            warmup_epochs=int(params.get("warmup_epochs", 10)),
            learning_rate=float(params.get("learning_rate", 0.01)),
            weight_decay=float(params.get("weight_decay", 0.0)),
            dropout=float(params.get("dropout", 0.0)),
            clustering_loss_gamma=float(params.get("clustering_loss_gamma", 10.0)),
            update_p_interval=int(params.get("update_p_interval", 5)),
            point_selection=str(params.get("point_selection", "core")),
            point_probability_threshold=float(params.get("point_probability_threshold", 0.5)),
            hdbscan_min_cluster_size=int(params.get("hdbscan_min_cluster_size", 10)),
            hdbscan_min_samples=int(params.get("hdbscan_min_samples", 5)),
            hdbscan_cluster_selection_method=str(params.get("hdbscan_cluster_selection_method", "eom")),
            mrd_k=int(params.get("mrd_k", params.get("hdbscan_min_samples", 5))),
            random_state=int(params.get("random_state", 42)),
            device=str(params.get("device", "auto")),
            verbose=bool(params.get("verbose", False)),
            log_interval=int(params.get("log_interval", 1)),
            evaluation_interval=int(params.get("evaluation_interval", 5)),
        ),
    )

    return {
        "assignments": output["assignments"],
        "features": output["embeddings"],
        "inertia": None,
        "metadata": {
            "name": "densgnn2",
            "input": "graph_topology_and_node_features",
            "uses_topology": True,
            "random_state": int(params.get("random_state", 42)),
            "encoder_hidden_channels": int(params.get("encoder_hidden_channels", 128)),
            "embedding_channels": int(params.get("embedding_channels", 32)),
            "epochs": int(params.get("epochs", 100)),
            "warmup_epochs": int(params.get("warmup_epochs", 10)),
            "learning_rate": float(params.get("learning_rate", 0.01)),
            "weight_decay": float(params.get("weight_decay", 0.0)),
            "dropout": float(params.get("dropout", 0.0)),
            "clustering_loss_gamma": float(params.get("clustering_loss_gamma", 10.0)),
            "update_p_interval": int(params.get("update_p_interval", 5)),
            "point_selection": str(params.get("point_selection", "core")),
            "point_probability_threshold": float(params.get("point_probability_threshold", 0.5)),
            "hdbscan_min_cluster_size": int(params.get("hdbscan_min_cluster_size", 10)),
            "hdbscan_min_samples": int(params.get("hdbscan_min_samples", 5)),
            "hdbscan_cluster_selection_method": str(params.get("hdbscan_cluster_selection_method", "eom")),
            "mrd_k": int(params.get("mrd_k", params.get("hdbscan_min_samples", 5))),
            "device": output["device"],
            "verbose": bool(params.get("verbose", False)),
            "log_interval": int(params.get("log_interval", 1)),
            "evaluation_interval": int(params.get("evaluation_interval", 5)),
            "evaluation_history": output["evaluation_history"],
            "hdbscan_runs": 1,
        },
        "extra_metrics": output["extra_metrics"],
    }
