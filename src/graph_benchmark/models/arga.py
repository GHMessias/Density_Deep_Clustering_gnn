from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import ARGA

from graph_benchmark.models.gae import TwoLayerGCNEncoder, resolve_torch_device


@dataclass(frozen=True)
class ARGATrainingConfig:
    input_channels: int
    hidden_channels: int
    embedding_channels: int
    discriminator_hidden_channels: int = 64
    epochs: int = 200
    learning_rate: float = 0.01
    discriminator_learning_rate: float = 0.001
    weight_decay: float = 0.0
    dropout: float = 0.0
    reg_loss_weight: float = 1.0
    discriminator_steps: int = 1
    device: str = "auto"


class ARGADiscriminator(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


def train_arga_embeddings(
    data: Data,
    config: ARGATrainingConfig,
) -> dict[str, Any]:
    if config.epochs < 1:
        raise ValueError("ARGA training requires 'epochs' to be at least 1.")

    device = resolve_torch_device(config.device)
    model = ARGA(
        encoder=TwoLayerGCNEncoder(
            input_channels=config.input_channels,
            hidden_channels=config.hidden_channels,
            embedding_channels=config.embedding_channels,
            dropout=config.dropout,
        ),
        discriminator=ARGADiscriminator(
            input_channels=config.embedding_channels,
            hidden_channels=config.discriminator_hidden_channels,
            dropout=config.dropout,
        ),
    ).to(device)
    encoder_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    discriminator_optimizer = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=config.discriminator_learning_rate,
        weight_decay=config.weight_decay,
    )
    working_data = data.clone().to(device)

    loss_history: list[dict[str, float]] = []

    for _epoch in range(config.epochs):
        model.train()

        encoder_optimizer.zero_grad()
        latent_embeddings = model.encode(working_data.x.float(), working_data.edge_index)
        reconstruction_loss = model.recon_loss(latent_embeddings, working_data.edge_index)
        regularization_loss = model.reg_loss(latent_embeddings)
        total_encoder_loss = reconstruction_loss + config.reg_loss_weight * regularization_loss
        total_encoder_loss.backward()
        encoder_optimizer.step()

        discriminator_loss_value = 0.0
        for _ in range(max(1, config.discriminator_steps)):
            discriminator_optimizer.zero_grad()
            latent_embeddings = model.encode(working_data.x.float(), working_data.edge_index)
            discriminator_loss = model.discriminator_loss(latent_embeddings)
            discriminator_loss.backward()
            discriminator_optimizer.step()
            discriminator_loss_value += float(discriminator_loss.item())

        loss_history.append(
            {
                "reconstruction_loss": float(reconstruction_loss.item()),
                "regularization_loss": float(regularization_loss.item()),
                "discriminator_loss": discriminator_loss_value / max(1, config.discriminator_steps),
                "total_loss": float(total_encoder_loss.item()),
            }
        )

    model.eval()
    with torch.no_grad():
        latent_embeddings = model.encode(working_data.x.float(), working_data.edge_index)

    final_losses = loss_history[-1]
    return {
        "embeddings": latent_embeddings.detach().cpu().numpy(),
        "final_reconstruction_loss": final_losses["reconstruction_loss"],
        "final_regularization_loss": final_losses["regularization_loss"],
        "final_discriminator_loss": final_losses["discriminator_loss"],
        "final_total_loss": final_losses["total_loss"],
        "loss_history": loss_history,
        "device": str(device),
    }
