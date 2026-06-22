from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec

from graph_benchmark.models.gae import resolve_torch_device


@dataclass(frozen=True)
class Node2VecTrainingConfig:
    embedding_channels: int = 128
    walk_length: int = 20
    context_size: int = 10
    walks_per_node: int = 10
    num_negative_samples: int = 1
    p: float = 1.0
    q: float = 1.0
    sparse: bool = True
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.01
    num_workers: int = 0
    device: str = "auto"


def train_node2vec_embeddings(
    data: Data,
    config: Node2VecTrainingConfig,
) -> dict[str, Any]:
    if config.epochs < 1:
        raise ValueError("Node2Vec training requires 'epochs' to be at least 1.")

    device = resolve_torch_device(config.device)
    model = Node2Vec(
        edge_index=data.edge_index,
        embedding_dim=config.embedding_channels,
        walk_length=config.walk_length,
        context_size=config.context_size,
        walks_per_node=config.walks_per_node,
        p=config.p,
        q=config.q,
        num_negative_samples=config.num_negative_samples,
        sparse=config.sparse,
        num_nodes=int(data.num_nodes),
    ).to(device)
    loader = model.loader(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    optimizer: torch.optim.Optimizer
    if config.sparse:
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    loss_history: list[float] = []

    for _epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        loss_history.append(epoch_loss / max(1, num_batches))

    model.eval()
    with torch.no_grad():
        embeddings = model().detach().cpu().numpy()

    return {
        "embeddings": embeddings,
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "device": str(device),
    }
