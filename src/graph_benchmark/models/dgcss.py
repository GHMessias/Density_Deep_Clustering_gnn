from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GATConv

from graph_benchmark.registry import SEED_SELECTOR_REGISTRY
from graph_benchmark.utils.clustering_loss import (
    kl_clustering_loss,
    student_t_soft_assignments,
    target_distribution,
)
from graph_benchmark.utils.graph_ops import build_t_hop_transition_graph
from graph_benchmark.utils.graph_ops import compute_clustering_accuracy, compute_modularity
from graph_benchmark.models.gae import resolve_torch_device


@dataclass(frozen=True)
class DGCSSConfig:
    input_channels: int
    hidden_channels: int
    embedding_channels: int
    n_clusters: int
    epochs: int = 400
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    dropout: float = 0.2
    attention_heads: int = 16
    transition_hops: int = 3
    clustering_loss_gamma: float = 20.0
    p_update_interval: int = 10
    centroid_update_step_size: float = 0.1
    reselect_centroids: bool = False
    reselect_patience: int = 100
    seed_selector: str = "betweenness_centrality"
    random_state: int = 42
    device: str = "auto"


class DGCSSGATEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        embedding_channels: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(
            input_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=False,
            edge_dim=1,
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            embedding_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=False,
            edge_dim=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x


def _select_initial_centroids(
    data: Data,
    embeddings: torch.Tensor,
    config: DGCSSConfig,
) -> tuple[list[int], torch.Tensor]:
    selector = SEED_SELECTOR_REGISTRY.get(config.seed_selector)
    selected_nodes, centroid_values = selector(
        data=data,
        embeddings=embeddings.detach().cpu().numpy(),
        n_seeds=config.n_clusters,
        random_state=config.random_state,
    )
    centroids = torch.tensor(
        np.asarray(centroid_values),
        dtype=embeddings.dtype,
        device=embeddings.device,
        requires_grad=True,
    )
    return selected_nodes, centroids


def train_dgcss(
    data: Data,
    labels: np.ndarray | None,
    config: DGCSSConfig,
) -> dict[str, Any]:
    if config.epochs < 1:
        raise ValueError("DGCSS requires 'epochs' to be at least 1.")

    device = resolve_torch_device(config.device)
    transition_edge_index, transition_edge_attr = build_t_hop_transition_graph(data, hops=config.transition_hops)
    working_data = data.clone().to(device)
    transition_edge_index = transition_edge_index.to(device)
    transition_edge_attr = transition_edge_attr.to(device)

    model = GAE(
        DGCSSGATEncoder(
            input_channels=config.input_channels,
            hidden_channels=config.hidden_channels,
            embedding_channels=config.embedding_channels,
            heads=config.attention_heads,
            dropout=config.dropout,
        )
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)

    centroids: torch.Tensor | None = None
    seed_node_ids: list[int] = []
    target_assignments: torch.Tensor | None = None
    loss_history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    best_total_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        should_reselect = centroids is None or (
            config.reselect_centroids and epochs_without_improvement >= config.reselect_patience
        )

        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(
            working_data.x.float(),
            transition_edge_index,
            transition_edge_attr,
        )

        if should_reselect:
            seed_node_ids, centroids = _select_initial_centroids(working_data, embeddings, config)
            target_assignments = None
            epochs_without_improvement = 0

        assert centroids is not None
        assignments = student_t_soft_assignments(embeddings, centroids)
        if target_assignments is None or epoch % config.p_update_interval == 0:
            target_assignments = target_distribution(assignments)

        clustering_loss = kl_clustering_loss(assignments, target_assignments)
        reconstruction_loss = model.recon_loss(embeddings, working_data.edge_index)
        total_loss = reconstruction_loss + config.clustering_loss_gamma * clustering_loss
        total_loss.backward()

        if centroids.grad is not None:
            with torch.no_grad():
                centroids = (
                    centroids - config.centroid_update_step_size * centroids.grad
                ).detach().requires_grad_(True)

        optimizer.step()
        scheduler.step()

        total_loss_value = float(total_loss.item())
        loss_history.append(
            {
                "epoch": float(epoch),
                "total_loss": total_loss_value,
                "reconstruction_loss": float(reconstruction_loss.item()),
                "clustering_loss": float(clustering_loss.item()),
            }
        )

        if total_loss_value < best_total_loss:
            best_total_loss = total_loss_value
            epochs_without_improvement = 0
            best_state = {
                "model": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "centroids": centroids.detach().cpu().clone(),
                "seed_node_ids": list(seed_node_ids),
            }
        else:
            epochs_without_improvement += 1

    if best_state is None:
        raise RuntimeError("DGCSS training did not produce a valid state.")

    model.load_state_dict(best_state["model"])
    centroids = best_state["centroids"].to(device).detach().requires_grad_(False)
    model.eval()

    with torch.no_grad():
        embeddings = model.encode(
            working_data.x.float(),
            transition_edge_index,
            transition_edge_attr,
        )
        assignments = student_t_soft_assignments(embeddings, centroids)
        hard_assignments = assignments.argmax(dim=1).detach().cpu().numpy()

    extra_metrics: dict[str, float] = {
        "reconstruction_loss": loss_history[-1]["reconstruction_loss"],
        "clustering_loss": loss_history[-1]["clustering_loss"],
        "total_loss": loss_history[-1]["total_loss"],
    }

    if labels is not None:
        extra_metrics["clustering_accuracy"] = compute_clustering_accuracy(labels, hard_assignments)

    extra_metrics["modularity"] = compute_modularity(data, hard_assignments)

    return {
        "embeddings": embeddings.detach().cpu().numpy(),
        "assignments": hard_assignments,
        "seed_node_ids": best_state["seed_node_ids"],
        "loss_history": loss_history,
        "extra_metrics": extra_metrics,
        "device": str(device),
    }
