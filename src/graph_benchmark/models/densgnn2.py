from __future__ import annotations

"""Variant of DensGNN that freezes the initial HDBSCAN partition.

The original DensGNN refreshes HDBSCAN whenever the target distribution P is
recomputed. DensGNN2 keeps exactly one HDBSCAN run after warm-up and only
updates P from the current Q afterwards.
"""

from typing import Any

import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GAE

from graph_benchmark.models.densgnn import (
    DensGNNClusterState,
    DensGNNConfig,
    _build_cluster_state,
    _compute_densgnn_assignments,
    _log_hdbscan_refresh,
    _should_run_evaluation,
)
from graph_benchmark.models.gae import TwoLayerGCNEncoder, resolve_torch_device
from graph_benchmark.utils.clustering_loss import kl_clustering_loss, target_distribution
from graph_benchmark.utils.graph_ops import compute_clustering_accuracy, compute_modularity


def train_densgnn2(
    data: Data,
    labels: np.ndarray | None,
    config: DensGNNConfig,
) -> dict[str, Any]:
    if config.epochs < 1:
        raise ValueError("DensGNN2 requires 'epochs' to be at least 1.")

    device = resolve_torch_device(config.device)
    model = GAE(
        encoder=TwoLayerGCNEncoder(
            input_channels=config.input_channels,
            hidden_channels=config.hidden_channels,
            embedding_channels=config.embedding_channels,
            dropout=config.dropout,
        )
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    working_data = data.clone().to(device)

    cluster_state: DensGNNClusterState | None = None
    target_assignments: torch.Tensor | None = None
    loss_history: list[dict[str, float]] = []
    evaluation_history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    best_total_loss = float("inf")
    best_eval_nmi = float("-inf")
    best_eval_epoch: int | None = None
    best_eval_cluster_count: int | None = None

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(working_data.x.float(), working_data.edge_index)
        reconstruction_loss = model.recon_loss(embeddings, working_data.edge_index)
        clustering_loss = torch.zeros((), device=device, dtype=embeddings.dtype)
        soft_assignments: torch.Tensor | None = None

        if epoch >= config.warmup_epochs:
            if cluster_state is None:
                cluster_state = _build_cluster_state(embeddings, config)
                target_assignments = None
                if config.verbose:
                    _log_hdbscan_refresh(epoch, cluster_state)

            if cluster_state is not None:
                soft_assignments = _compute_densgnn_assignments(embeddings, cluster_state, config)
                should_update_targets = (
                    target_assignments is None
                    or (epoch - config.warmup_epochs) % config.update_p_interval == 0
                )
                if should_update_targets:
                    target_assignments = target_distribution(soft_assignments).detach()
                clustering_loss = kl_clustering_loss(soft_assignments, target_assignments)

        total_loss = reconstruction_loss + config.clustering_loss_gamma * clustering_loss
        total_loss.backward()
        optimizer.step()

        reconstruction_loss_value = float(reconstruction_loss.item())
        clustering_loss_value = float(clustering_loss.item())
        total_loss_value = float(total_loss.item())
        selected_clusters = len(cluster_state.supports) if cluster_state is not None else 0

        loss_history.append(
            {
                "epoch": float(epoch + 1),
                "reconstruction_loss": reconstruction_loss_value,
                "clustering_loss": clustering_loss_value,
                "total_loss": total_loss_value,
                "selected_clusters": float(selected_clusters),
            }
        )

        if total_loss_value < best_total_loss:
            best_total_loss = total_loss_value
            if labels is None or best_eval_epoch is None:
                best_state = {
                    "model": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    "cluster_state": cluster_state,
                }

        if (
            labels is not None
            and soft_assignments is not None
            and cluster_state is not None
            and _should_run_evaluation(epoch, config)
        ):
            eval_assignments = soft_assignments.argmax(dim=1).detach().cpu().numpy()
            eval_nmi = float(normalized_mutual_info_score(labels, eval_assignments))
            evaluation_entry = {
                "epoch": float(epoch + 1),
                "nmi": eval_nmi,
                "num_clusters": float(len(np.unique(eval_assignments))),
                "noise_ratio": cluster_state.noise_ratio,
            }
            evaluation_history.append(evaluation_entry)

            if config.verbose:
                print(
                    "[densgnn2] "
                    f"epoch={epoch + 1:03d}/{config.epochs:03d} "
                    f"eval_nmi={eval_nmi:.6f} "
                    f"eval_clusters={int(evaluation_entry['num_clusters'])}"
                )

            if eval_nmi > best_eval_nmi:
                best_eval_nmi = eval_nmi
                best_eval_epoch = epoch + 1
                best_eval_cluster_count = int(evaluation_entry["num_clusters"])
                best_state = {
                    "model": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    "cluster_state": cluster_state,
                    "eval_assignments": eval_assignments.copy(),
                    "eval_noise_ratio": cluster_state.noise_ratio,
                    "eval_support_points": float(cluster_state.support_points),
                }

        if config.verbose and (
            epoch == 0
            or (epoch + 1) % max(1, config.log_interval) == 0
            or epoch + 1 == config.epochs
        ):
            print(
                "[densgnn2] "
                f"epoch={epoch + 1:03d}/{config.epochs:03d} "
                f"recon_loss={reconstruction_loss_value:.6f} "
                f"cluster_loss={clustering_loss_value:.6f} "
                f"total_loss={total_loss_value:.6f} "
                f"selected_clusters={selected_clusters}"
            )

    if best_state is None:
        raise RuntimeError("DensGNN2 training did not produce a valid state.")

    model.load_state_dict(best_state["model"])
    model.eval()

    with torch.no_grad():
        embeddings = model.encode(working_data.x.float(), working_data.edge_index)
        reconstruction_loss = model.recon_loss(embeddings, working_data.edge_index)

    final_cluster_state = best_state.get("cluster_state")
    if final_cluster_state is None:
        hard_assignments = np.zeros(data.num_nodes, dtype=np.int64)
        final_clustering_loss_value = 0.0
        final_num_clusters = 1
        final_noise_ratio = 1.0
        final_support_points = 0
    else:
        final_soft_assignments = _compute_densgnn_assignments(embeddings, final_cluster_state, config)
        final_targets = target_distribution(final_soft_assignments)
        final_clustering_loss_value = float(
            kl_clustering_loss(final_soft_assignments, final_targets).item()
        )
        hard_assignments = final_soft_assignments.argmax(dim=1).detach().cpu().numpy()
        final_num_clusters = len(final_cluster_state.supports)
        final_noise_ratio = final_cluster_state.noise_ratio
        final_support_points = final_cluster_state.support_points

    if labels is not None and best_eval_epoch is not None and "eval_assignments" in best_state:
        hard_assignments = np.asarray(best_state["eval_assignments"], dtype=np.int64)
        final_num_clusters = int(np.unique(hard_assignments).size)
        final_noise_ratio = float(best_state.get("eval_noise_ratio", final_noise_ratio))
        final_support_points = float(best_state.get("eval_support_points", final_support_points))

    extra_metrics: dict[str, float] = {
        "reconstruction_loss": float(reconstruction_loss.item()),
        "clustering_loss": final_clustering_loss_value,
        "total_loss": float(reconstruction_loss.item() + config.clustering_loss_gamma * final_clustering_loss_value),
        "hdbscan_num_clusters": float(final_num_clusters),
        "hdbscan_noise_ratio": float(final_noise_ratio),
        "support_points": float(final_support_points),
    }

    if labels is not None:
        extra_metrics["clustering_accuracy"] = compute_clustering_accuracy(labels, hard_assignments)
        if best_eval_epoch is not None:
            extra_metrics["best_eval_nmi"] = float(best_eval_nmi)
            extra_metrics["best_eval_epoch"] = float(best_eval_epoch)
        if best_eval_cluster_count is not None:
            extra_metrics["best_eval_num_clusters"] = float(best_eval_cluster_count)

    extra_metrics["modularity"] = compute_modularity(data, hard_assignments)

    return {
        "embeddings": embeddings.detach().cpu().numpy(),
        "assignments": hard_assignments,
        "loss_history": loss_history,
        "evaluation_history": evaluation_history,
        "extra_metrics": extra_metrics,
        "device": str(device),
    }
