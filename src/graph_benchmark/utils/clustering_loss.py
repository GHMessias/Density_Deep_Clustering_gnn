from __future__ import annotations

import torch
import torch.nn.functional as F


def student_t_soft_assignments(
    embeddings: torch.Tensor,
    centroids: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    diff = embeddings.unsqueeze(1) - centroids.unsqueeze(0)
    squared_distances = torch.sum(diff.pow(2), dim=2)
    numerator = (1.0 + squared_distances / alpha).pow(-(alpha + 1.0) / 2.0)
    return numerator / numerator.sum(dim=1, keepdim=True)


def target_distribution(assignments: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    frequencies = assignments.sum(dim=0, keepdim=True)
    weights = assignments.pow(2) / (frequencies + eps)
    return weights / weights.sum(dim=1, keepdim=True)


def kl_clustering_loss(
    assignments: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return F.kl_div(assignments.clamp_min(1e-12).log(), targets.detach(), reduction="batchmean")
