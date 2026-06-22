from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv


@dataclass(frozen=True)
class GAETrainingConfig:
    input_channels: int
    hidden_channels: int
    embedding_channels: int
    epochs: int = 200
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    dropout: float = 0.0
    device: str = "auto"


class TwoLayerGCNEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        embedding_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def resolve_torch_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(requested_device)


def train_gae_embeddings(
    data: Data,
    config: GAETrainingConfig,
) -> dict[str, Any]:
    if config.epochs < 1:
        raise ValueError("GAE training requires 'epochs' to be at least 1.")

    device = resolve_torch_device(config.device)
    model = GAE(
        TwoLayerGCNEncoder(
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
    loss_history: list[float] = []

    for _epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        latent_embeddings = model.encode(working_data.x, working_data.edge_index)
        loss = model.recon_loss(latent_embeddings, working_data.edge_index)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        latent_embeddings = model.encode(working_data.x, working_data.edge_index)

    return {
        "embeddings": latent_embeddings.detach().cpu().numpy(),
        "final_reconstruction_loss": loss_history[-1],
        "loss_history": loss_history,
        "device": str(device),
    }
