from __future__ import annotations

from typing import Any

from graph_benchmark.models.dgcss import DGCSSConfig, train_dgcss
from graph_benchmark.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("dgcss")
def run_dgcss(
    dataset_bundle: dict[str, Any],
    algorithm_config: dict[str, Any],
) -> dict[str, Any]:
    dataset = dataset_bundle["dataset"]
    data = dataset_bundle["data"]
    params = algorithm_config.get("params", {})
    labels = data.y.detach().cpu().numpy() if getattr(data, "y", None) is not None else None

    n_clusters_config = params.get("n_clusters", "auto")
    n_clusters = int(dataset.num_classes) if n_clusters_config == "auto" else int(n_clusters_config)

    output = train_dgcss(
        data,
        labels=labels,
        config=DGCSSConfig(
            input_channels=int(dataset.num_features),
            hidden_channels=int(params.get("encoder_hidden_channels", 64)),
            embedding_channels=int(params.get("embedding_channels", 16)),
            n_clusters=n_clusters,
            epochs=int(params.get("epochs", 400)),
            learning_rate=float(params.get("learning_rate", 5e-5)),
            weight_decay=float(params.get("weight_decay", 0.0)),
            dropout=float(params.get("dropout", 0.2)),
            attention_heads=int(params.get("attention_heads", 16)),
            transition_hops=int(params.get("transition_hops", 3)),
            clustering_loss_gamma=float(params.get("clustering_loss_gamma", 20.0)),
            p_update_interval=int(params.get("p_update_interval", 10)),
            centroid_update_step_size=float(params.get("centroid_update_step_size", 0.1)),
            reselect_centroids=bool(params.get("reselect_centroids", False)),
            reselect_patience=int(params.get("reselect_patience", 100)),
            seed_selector=str(params.get("seed_selector", "betweenness_centrality")),
            random_state=int(params.get("random_state", 42)),
            device=str(params.get("device", "auto")),
        ),
    )

    return {
        "assignments": output["assignments"],
        "features": output["embeddings"],
        "inertia": None,
        "metadata": {
            "name": "dgcss",
            "input": "graph_topology_and_node_features",
            "uses_topology": True,
            "n_clusters": n_clusters,
            "seed_selector": str(params.get("seed_selector", "betweenness_centrality")),
            "random_state": int(params.get("random_state", 42)),
            "encoder_hidden_channels": int(params.get("encoder_hidden_channels", 64)),
            "embedding_channels": int(params.get("embedding_channels", 16)),
            "epochs": int(params.get("epochs", 400)),
            "learning_rate": float(params.get("learning_rate", 5e-5)),
            "weight_decay": float(params.get("weight_decay", 0.0)),
            "dropout": float(params.get("dropout", 0.2)),
            "attention_heads": int(params.get("attention_heads", 16)),
            "transition_hops": int(params.get("transition_hops", 3)),
            "clustering_loss_gamma": float(params.get("clustering_loss_gamma", 20.0)),
            "p_update_interval": int(params.get("p_update_interval", 10)),
            "centroid_update_step_size": float(params.get("centroid_update_step_size", 0.1)),
            "reselect_centroids": bool(params.get("reselect_centroids", False)),
            "reselect_patience": int(params.get("reselect_patience", 100)),
            "seed_node_ids": output["seed_node_ids"],
            "device": output["device"],
        },
        "extra_metrics": output["extra_metrics"],
    }
