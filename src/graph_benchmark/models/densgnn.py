from __future__ import annotations

"""Implementation of DensGNN.

DensGNN is a density-guided deep graph clustering baseline built on top of a
Graph Autoencoder (GAE) whose encoder is a two-layer GCN and whose decoder is
the standard inner-product decoder from PyG's ``GAE`` wrapper.

Training follows an alternating scheme:

1. Warm-up the autoencoder using only graph reconstruction.
2. Run HDBSCAN on the current latent embeddings.
3. Select support nodes from each cluster (currently using core-like points by
   default).
4. Compute DensGNN soft assignments Q from average MRD to those supports.
5. Refresh the target distribution P every ``update_p_interval`` epochs and
   keep P fixed between refreshes, while Q is recomputed every epoch.

This keeps the clustering signal tied to dense support regions instead of
centroids.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GAE

from graph_benchmark.models.gae import TwoLayerGCNEncoder, resolve_torch_device
from graph_benchmark.utils.clustering_loss import kl_clustering_loss, target_distribution
from graph_benchmark.utils.densgnn_loss import (
    ClusterSupport,
    inverse_distance_assignments,
    support_cluster_distances,
)
from graph_benchmark.utils.graph_ops import compute_clustering_accuracy, compute_modularity


@dataclass(frozen=True)
class DensGNNConfig:
    input_channels: int
    hidden_channels: int
    embedding_channels: int
    epochs: int = 100
    warmup_epochs: int = 10
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    dropout: float = 0.0
    clustering_loss_gamma: float = 10.0
    update_p_interval: int = 5
    point_selection: str = "core"
    point_probability_threshold: float = 0.5
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5
    hdbscan_cluster_selection_method: str = "eom"
    mrd_k: int = 5
    random_state: int = 42
    device: str = "auto"
    verbose: bool = False
    log_interval: int = 1
    evaluation_interval: int = 5


@dataclass(frozen=True)
class DensGNNClusterState:
    """Frozen HDBSCAN output reused across several training epochs."""

    supports: list[ClusterSupport]
    raw_labels: np.ndarray
    raw_probabilities: np.ndarray
    noise_ratio: float
    support_points: int


def _select_support_node_ids(
    cluster_node_ids: np.ndarray,
    cluster_probabilities: np.ndarray,
    point_selection: str,
    probability_threshold: float,
) -> np.ndarray:
    """Choose support points from one HDBSCAN cluster.

    ``sklearn.cluster.HDBSCAN`` does not expose explicit core/border tags. To
    keep the implementation compatible with the requested API, we approximate:

    - ``core``: members whose HDBSCAN membership strength is at least the
      configured threshold;
    - ``border``: non-noise members below that threshold.

    When the requested subset is empty, we fall back to a small non-empty set so
    the loss remains computable.
    """

    if point_selection not in {"core", "border"}:
        raise ValueError("point_selection must be either 'core' or 'border'.")

    if point_selection == "core":
        mask = cluster_probabilities >= probability_threshold
        if np.any(mask):
            return cluster_node_ids[mask]

        fallback_count = min(max(1, cluster_node_ids.size // 4), cluster_node_ids.size)
        top_indices = np.argsort(cluster_probabilities)[-fallback_count:]
        return cluster_node_ids[top_indices]

    mask = cluster_probabilities < probability_threshold
    if np.any(mask):
        return cluster_node_ids[mask]

    fallback_count = min(max(1, cluster_node_ids.size // 4), cluster_node_ids.size)
    bottom_indices = np.argsort(cluster_probabilities)[:fallback_count]
    return cluster_node_ids[bottom_indices]


def _build_cluster_state(
    embeddings: torch.Tensor,
    config: DensGNNConfig,
) -> DensGNNClusterState | None:
    """Run HDBSCAN on detached embeddings and prepare DensGNN support sets."""

    embeddings_np = embeddings.detach().cpu().numpy()
    clusterer = HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        cluster_selection_method=config.hdbscan_cluster_selection_method,
        copy=True,
    )
    clusterer.fit(embeddings_np)

    raw_labels = clusterer.labels_.astype(np.int64)
    raw_probabilities = clusterer.probabilities_.astype(np.float32)
    supports: list[ClusterSupport] = []

    for label in sorted(int(value) for value in np.unique(raw_labels) if value >= 0):
        cluster_node_ids = np.flatnonzero(raw_labels == label)
        cluster_probabilities = raw_probabilities[cluster_node_ids]
        support_node_ids = _select_support_node_ids(
            cluster_node_ids=cluster_node_ids,
            cluster_probabilities=cluster_probabilities,
            point_selection=config.point_selection,
            probability_threshold=config.point_probability_threshold,
        )
        if support_node_ids.size == 0:
            continue

        supports.append(
            ClusterSupport(
                label=label,
                node_ids=torch.as_tensor(support_node_ids, dtype=torch.long, device=embeddings.device),
            )
        )

    if len(supports) < 2:
        return None

    return DensGNNClusterState(
        supports=supports,
        raw_labels=raw_labels,
        raw_probabilities=raw_probabilities,
        noise_ratio=float(np.mean(raw_labels == -1)),
        support_points=int(sum(int(support.node_ids.numel()) for support in supports)),
    )


def _compute_densgnn_assignments(
    embeddings: torch.Tensor,
    cluster_state: DensGNNClusterState,
    config: DensGNNConfig,
) -> torch.Tensor:
    """Compute DensGNN soft assignments Q for the current embeddings."""

    cluster_distances = support_cluster_distances(
        embeddings=embeddings,
        supports=cluster_state.supports,
        mrd_k=config.mrd_k,
    )
    return inverse_distance_assignments(cluster_distances)


def _log_hdbscan_refresh(epoch: int, cluster_state: DensGNNClusterState | None) -> None:
    if cluster_state is None:
        print(f"[densgnn] epoch={epoch:03d} hdbscan_update=invalid_clustering")
        return

    print(
        "[densgnn] "
        f"epoch={epoch:03d} "
        f"hdbscan_clusters={len(cluster_state.supports)} "
        f"support_points={cluster_state.support_points} "
        f"noise_ratio={cluster_state.noise_ratio:.4f}"
    )


def _should_run_evaluation(epoch: int, config: DensGNNConfig) -> bool:
    """Return whether this epoch should trigger a checkpoint evaluation."""

    interval = max(1, config.evaluation_interval)
    return (epoch + 1) % interval == 0 or (epoch + 1) == config.epochs


def _format_cluster_distribution(assignments: np.ndarray) -> str:
    unique_labels, counts = np.unique(assignments, return_counts=True)
    parts = [f"{int(label)}:{int(count)}" for label, count in zip(unique_labels, counts, strict=True)]
    return "{" + ", ".join(parts) + "}"


def train_densgnn(
    data: Data,
    labels: np.ndarray | None,
    config: DensGNNConfig,
) -> dict[str, Any]:
    """Train DensGNN and return final assignments plus benchmark metadata."""

    if config.epochs < 1:
        raise ValueError("DensGNN requires 'epochs' to be at least 1.")

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
            should_refresh_targets = (
                cluster_state is None
                or target_assignments is None
                or (epoch - config.warmup_epochs) % config.update_p_interval == 0
            )
            if should_refresh_targets:
                refreshed_state = _build_cluster_state(embeddings, config)
                if refreshed_state is not None:
                    cluster_state = refreshed_state
                    target_assignments = None
                if config.verbose:
                    _log_hdbscan_refresh(epoch, cluster_state)

            if cluster_state is not None:
                soft_assignments = _compute_densgnn_assignments(embeddings, cluster_state, config)
                if target_assignments is None:
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
                    "[densgnn] "
                    f"epoch={epoch + 1:03d}/{config.epochs:03d} "
                    f"eval_nmi={eval_nmi:.6f} "
                    f"eval_clusters={int(evaluation_entry['num_clusters'])} "
                    f"cluster_sizes={_format_cluster_distribution(eval_assignments)} "
                    f"hdbscan_outliers={int(np.sum(cluster_state.raw_labels == -1))}"
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
                "[densgnn] "
                f"epoch={epoch + 1:03d}/{config.epochs:03d} "
                f"recon_loss={reconstruction_loss_value:.6f} "
                f"cluster_loss={clustering_loss_value:.6f} "
                f"total_loss={total_loss_value:.6f} "
                f"selected_clusters={selected_clusters}"
            )

    if best_state is None:
        raise RuntimeError("DensGNN training did not produce a valid state.")

    model.load_state_dict(best_state["model"])
    model.eval()

    with torch.no_grad():
        embeddings = model.encode(working_data.x.float(), working_data.edge_index)
        reconstruction_loss = model.recon_loss(embeddings, working_data.edge_index)

    final_cluster_state = _build_cluster_state(embeddings, config)
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
