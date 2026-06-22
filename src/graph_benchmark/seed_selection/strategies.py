from __future__ import annotations

import random
from typing import Any

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.utils import to_networkx

from graph_benchmark.registry import SEED_SELECTOR_REGISTRY


def _select_topk_by_score(
    scores: dict[int, float],
    n_seeds: int,
) -> list[int]:
    ordered_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [node for node, _score in ordered_nodes[:n_seeds]]


@SEED_SELECTOR_REGISTRY.register("random")
def select_random_seeds(
    data: Any,
    embeddings: np.ndarray,
    n_seeds: int,
    random_state: int = 42,
) -> tuple[list[int], np.ndarray]:
    if n_seeds > data.num_nodes:
        raise ValueError("Number of requested seeds cannot be greater than the number of nodes.")

    rng = random.Random(random_state)
    selected_nodes = rng.sample(range(data.num_nodes), n_seeds)
    return selected_nodes, embeddings[selected_nodes]


@SEED_SELECTOR_REGISTRY.register("betweenness_centrality")
def select_betweenness_centrality_seeds(
    data: Any,
    embeddings: np.ndarray,
    n_seeds: int,
    random_state: int = 42,
) -> tuple[list[int], np.ndarray]:
    del random_state
    graph = nx.Graph(to_networkx(data, node_attrs=["x"]))
    centrality = nx.betweenness_centrality(graph)
    selected_nodes = _select_topk_by_score(centrality, n_seeds)
    return selected_nodes, embeddings[selected_nodes]


@SEED_SELECTOR_REGISTRY.register("closeness_centrality")
def select_closeness_centrality_seeds(
    data: Any,
    embeddings: np.ndarray,
    n_seeds: int,
    random_state: int = 42,
) -> tuple[list[int], np.ndarray]:
    del random_state
    graph = nx.Graph(to_networkx(data, node_attrs=["x"]))
    centrality = nx.closeness_centrality(graph)
    selected_nodes = _select_topk_by_score(centrality, n_seeds)
    return selected_nodes, embeddings[selected_nodes]


@SEED_SELECTOR_REGISTRY.register("kmeans")
def select_kmeans_seeds(
    data: Any,
    embeddings: np.ndarray,
    n_seeds: int,
    random_state: int = 42,
) -> tuple[list[int], np.ndarray]:
    del data
    model = KMeans(n_clusters=n_seeds, random_state=random_state, n_init=20)
    model.fit(embeddings)

    centroids = model.cluster_centers_
    distances = ((embeddings[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    nearest_node_indices = []

    for cluster_index in range(n_seeds):
        ordered_nodes = np.argsort(distances[:, cluster_index])
        for node_index in ordered_nodes:
            node_id = int(node_index)
            if node_id not in nearest_node_indices:
                nearest_node_indices.append(node_id)
                break

    if len(nearest_node_indices) != n_seeds:
        raise RuntimeError("KMeans seed selection could not find a unique representative node for each centroid.")

    return nearest_node_indices, embeddings[nearest_node_indices]
