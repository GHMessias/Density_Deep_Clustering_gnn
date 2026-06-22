from __future__ import annotations

from typing import Any

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

from graph_benchmark.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("ollivier_ricci_community")
def run_ollivier_ricci_community(
    dataset_bundle: dict[str, Any],
    algorithm_config: dict[str, Any],
) -> dict[str, Any]:
    data = dataset_bundle["data"]
    params = algorithm_config.get("params", {})
    ricci_params = params.get("ricci", {})
    flow_params = params.get("flow", {})
    community_params = params.get("community", {})

    graph = to_networkx(data, to_undirected=True, remove_self_loops=True)
    assignments = np.full(data.num_nodes, -1, dtype=int)
    component_summaries: list[dict[str, int | float | str]] = []
    next_cluster_id = 0

    for component_index, component_nodes in enumerate(nx.connected_components(graph)):
        subgraph = graph.subgraph(component_nodes).copy()

        if subgraph.number_of_nodes() < 3 or subgraph.number_of_edges() == 0:
            for node in subgraph.nodes():
                assignments[node] = next_cluster_id

            component_summaries.append(
                {
                    "component_index": component_index,
                    "num_nodes": int(subgraph.number_of_nodes()),
                    "num_edges": int(subgraph.number_of_edges()),
                    "num_clusters": 1,
                    "status": "single_cluster_fallback",
                }
            )
            next_cluster_id += 1
            continue

        ricci_graph = OllivierRicci(
            subgraph,
            alpha=float(ricci_params.get("alpha", 0.5)),
            method=str(ricci_params.get("method", "OTDSinkhornMix")),
            proc=int(ricci_params.get("proc", 1)),
            verbose=str(ricci_params.get("verbose", "ERROR")),
        )
        ricci_graph.compute_ricci_flow(
            iterations=int(flow_params.get("iterations", 10)),
            step=float(flow_params.get("step", 1.0)),
            delta=float(flow_params.get("delta", 1e-4)),
        )

        try:
            cutoff, clustering = ricci_graph.ricci_community(
                cutoff_step=float(community_params.get("cutoff_step", 0.025)),
                drop_threshold=float(community_params.get("drop_threshold", 0.01)),
            )
            local_clusters = sorted(set(clustering.values()))
            local_to_global = {
                local_cluster: next_cluster_id + offset
                for offset, local_cluster in enumerate(local_clusters)
            }

            for node, local_cluster in clustering.items():
                assignments[node] = local_to_global[local_cluster]

            component_summaries.append(
                {
                    "component_index": component_index,
                    "num_nodes": int(subgraph.number_of_nodes()),
                    "num_edges": int(subgraph.number_of_edges()),
                    "num_clusters": int(len(local_clusters)),
                    "cutoff": float(cutoff),
                    "status": "ricci_community",
                }
            )
            next_cluster_id += len(local_clusters)
        except (AssertionError, IndexError, ValueError):
            for node in subgraph.nodes():
                assignments[node] = next_cluster_id

            component_summaries.append(
                {
                    "component_index": component_index,
                    "num_nodes": int(subgraph.number_of_nodes()),
                    "num_edges": int(subgraph.number_of_edges()),
                    "num_clusters": 1,
                    "status": "ricci_cutoff_fallback",
                }
            )
            next_cluster_id += 1

    if np.any(assignments < 0):
        raise RuntimeError("Ricci community clustering did not assign all nodes to a cluster.")

    detected_cutoffs = [
        summary["cutoff"]
        for summary in component_summaries
        if "cutoff" in summary
    ]

    return {
        "assignments": assignments,
        "features": None,
        "inertia": None,
        "metadata": {
            "name": "ollivier_ricci_community",
            "input": "graph_topology",
            "uses_topology": True,
            "alpha": float(ricci_params.get("alpha", 0.5)),
            "method": str(ricci_params.get("method", "OTDSinkhornMix")),
            "proc": int(ricci_params.get("proc", 1)),
            "flow_iterations": int(flow_params.get("iterations", 10)),
            "flow_step": float(flow_params.get("step", 1.0)),
            "flow_delta": float(flow_params.get("delta", 1e-4)),
            "cutoff_step": float(community_params.get("cutoff_step", 0.025)),
            "drop_threshold": float(community_params.get("drop_threshold", 0.01)),
            "num_connected_components": int(len(component_summaries)),
            "component_summaries": component_summaries,
        },
        "extra_metrics": {
            "ricci_num_connected_components": float(len(component_summaries)),
            "ricci_mean_cutoff": float(sum(detected_cutoffs) / len(detected_cutoffs)) if detected_cutoffs else 0.0,
        },
    }
