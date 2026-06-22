from __future__ import annotations

"""Loss utilities for DensGNN.

DensGNN replaces centroid-based cluster representations with support sets
obtained from HDBSCAN. Each cluster is represented by a subset of nodes
selected from the latent space. The clustering signal is then computed from the
average mutual reachability distance (MRD) between a node embedding and the
support nodes of every cluster.

This file keeps the math explicit because the implementation is meant to be
read, reviewed and iterated on as part of the benchmark project.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClusterSupport:
    """Support nodes used to represent one HDBSCAN cluster inside DensGNN.

    Parameters
    ----------
    label:
        Original cluster label returned by HDBSCAN.
    node_ids:
        Node ids chosen to represent the cluster in the loss.
    """

    label: int
    node_ids: torch.Tensor


def _kth_neighbor_distances(
    pairwise_distances: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Approximate the core distance used inside mutual reachability distance."""

    num_nodes = pairwise_distances.size(0)
    if num_nodes < 2:
        return torch.zeros(num_nodes, dtype=pairwise_distances.dtype, device=pairwise_distances.device)

    effective_k = max(1, min(k, num_nodes - 1))
    masked = pairwise_distances.clone()
    masked.fill_diagonal_(float("inf"))
    knn_distances, _ = torch.topk(masked, k=effective_k, dim=1, largest=False)
    return knn_distances[:, -1]


def support_cluster_distances(
    embeddings: torch.Tensor,
    supports: list[ClusterSupport],
    mrd_k: int,
) -> torch.Tensor:
    """Compute the DensGNN cluster distance matrix.

    For node i and cluster j, DensGNN uses:

        d_ij = (1 / |Omega_j|) * sum_{l in Omega_j} MRD(z_i, z_l)

    where Omega_j is the selected support set of cluster j and MRD is the
    mutual reachability distance in the latent space.
    """

    if not supports:
        raise ValueError("At least one support cluster is required.")

    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
    core_distances = _kth_neighbor_distances(pairwise_distances, k=mrd_k)
    node_core_distances = core_distances.unsqueeze(1)
    per_cluster_distances: list[torch.Tensor] = []

    for support in supports:
        point_to_support = pairwise_distances[:, support.node_ids]
        support_core_distances = core_distances[support.node_ids].unsqueeze(0)
        mrd = torch.maximum(point_to_support, node_core_distances)
        mrd = torch.maximum(mrd, support_core_distances)
        per_cluster_distances.append(mrd.mean(dim=1))

    return torch.stack(per_cluster_distances, dim=1)


def inverse_distance_assignments(
    cluster_distances: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert cluster distances into DensGNN soft assignments Q."""

    inverse_distances = 1.0 / cluster_distances.clamp_min(eps)
    return inverse_distances / inverse_distances.sum(dim=1, keepdim=True).clamp_min(eps)
