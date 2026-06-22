from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_scipy_sparse_matrix
import networkx as nx


def build_t_hop_transition_graph(
    data: Data,
    hops: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hops < 1:
        raise ValueError("'hops' must be at least 1.")

    adjacency = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    adjacency.data = np.ones_like(adjacency.data, dtype=np.float32)

    row_sums = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_row_sums = np.zeros_like(row_sums, dtype=np.float32)
    nonzero_mask = row_sums > 0
    inv_row_sums[nonzero_mask] = 1.0 / row_sums[nonzero_mask]

    transition = sp.diags(inv_row_sums) @ adjacency
    transition_sum = transition.copy()
    transition_power = transition.copy()

    for _ in range(2, hops + 1):
        transition_power = transition_power @ transition
        transition_sum = transition_sum + transition_power

    averaged_transition = (transition_sum / hops).tocoo()
    averaged_transition.eliminate_zeros()

    edge_index = torch.from_numpy(
        np.vstack((averaged_transition.row, averaged_transition.col))
    ).long()
    edge_weight = torch.from_numpy(averaged_transition.data.astype(np.float32)).view(-1, 1)
    return edge_index, edge_weight


def compute_modularity(data: Data, assignments: np.ndarray) -> float:
    graph = nx.Graph(to_networkx(data))
    communities = [set(np.where(assignments == cluster_id)[0]) for cluster_id in np.unique(assignments)]
    return float(nx.community.modularity(graph, communities))


def compute_clustering_accuracy(labels: np.ndarray, assignments: np.ndarray) -> float:
    labels = np.asarray(labels)
    assignments = np.asarray(assignments)
    size = int(max(assignments.max(), labels.max()) + 1)
    contingency = np.zeros((size, size), dtype=np.int64)

    for index in range(len(labels)):
        contingency[assignments[index], labels[index]] += 1

    from scipy.optimize import linear_sum_assignment

    row_indices, col_indices = linear_sum_assignment(-contingency)
    return float(contingency[row_indices, col_indices].sum() / len(labels))
