from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from graph_benchmark.registry import DATASET_REGISTRY
from graph_benchmark.utils.io import ensure_directory, resolve_project_path


@dataclass(frozen=True)
class SyntheticGraphDataset:
    name: str
    num_features: int
    num_classes: int


def _resolve_gencat_import() -> Any:
    import sys
    import types

    gencat_root = Path(__file__).resolve().parents[3] / "GenCAT"
    if str(gencat_root) not in sys.path:
        sys.path.insert(0, str(gencat_root))

    try:
        import gencat as gencat_module
    except ImportError as exc:  # pragma: no cover
        if exc.name != "powerlaw":
            raise ImportError(
                "Failed to import GenCAT. Install its Python dependencies before "
                "using the synthetic real-scenario benchmark."
            ) from exc

        # GenCAT relies on the third-party ``powerlaw`` package only to sample a
        # power-law-like degree sequence. For the benchmark loader we provide a
        # lightweight shim with the API shape expected by GenCAT, so the
        # synthetic dataset can still be generated inside this repository.
        fallback_module = types.ModuleType("powerlaw")

        class _PowerLawShim:
            def __init__(self, xmin: float = 1.0, parameters: list[float] | tuple[float, ...] | None = None) -> None:
                self.xmin = float(xmin)
                self.parameters = parameters or [3.0]

            def generate_random(self, size: int) -> np.ndarray:
                exponent = float(self.parameters[0]) if self.parameters else 3.0
                # The external ``powerlaw`` package used by GenCAT yields a
                # heavier-tailed degree proposal than a direct Pareto draw with
                # the same exponent. We therefore shift the shape down by one to
                # better match GenCAT's expected edge counts in practice.
                shape = max(1.01, exponent - 1.0)
                return self.xmin * (1.0 + np.random.pareto(shape, int(size)))

        fallback_module.Power_Law = _PowerLawShim
        sys.modules["powerlaw"] = fallback_module
        import gencat as gencat_module

    return gencat_module


def _build_ring_preference_matrix(
    num_classes: int,
    diagonal_preference: float,
    neighbor_preference: float,
) -> np.ndarray:
    if not 0.0 < diagonal_preference < 1.0:
        raise ValueError("diagonal_preference must be in (0, 1).")
    if not 0.0 <= neighbor_preference < 1.0:
        raise ValueError("neighbor_preference must be in [0, 1).")

    remaining = 1.0 - diagonal_preference - 2.0 * neighbor_preference
    if remaining < 0:
        raise ValueError("Invalid topology preferences: probabilities exceed 1.")

    background_count = max(1, num_classes - 3)
    background_value = remaining / background_count

    matrix = np.full((num_classes, num_classes), background_value, dtype=np.float32)
    for class_id in range(num_classes):
        matrix[class_id, class_id] = diagonal_preference
        matrix[class_id, (class_id - 1) % num_classes] = neighbor_preference
        matrix[class_id, (class_id + 1) % num_classes] = neighbor_preference
        matrix[class_id] /= matrix[class_id].sum()
    return matrix


def _build_preference_deviation_matrix(num_classes: int, deviation: float) -> np.ndarray:
    return np.full((num_classes, num_classes), deviation, dtype=np.float32)


def _build_attribute_correlation_matrix(
    num_features: int,
    num_classes: int,
    primary_strength: float,
    secondary_strength: float,
    background_strength: float,
) -> np.ndarray:
    matrix = np.full((num_features, num_classes), background_strength, dtype=np.float32)
    block_size = max(1, num_features // num_classes)

    for feature_id in range(num_features):
        primary_class = min(feature_id // block_size, num_classes - 1)
        secondary_class = (primary_class + 1) % num_classes
        matrix[feature_id, primary_class] = primary_strength
        matrix[feature_id, secondary_class] = secondary_strength
        matrix[feature_id] /= matrix[feature_id].sum()

    return matrix


def _match_target_edge_count(
    adjacency: Any,
    labels: np.ndarray,
    target_edges: int,
    preference_matrix: np.ndarray,
    seed: int,
) -> Any:
    """Adjust the synthetic graph to stay close to the requested edge budget.

    GenCAT's internal degree-generation loop is stochastic and may undershoot
    the desired number of edges. For the benchmark we lightly post-process the
    graph so the final scenario matches the intended "about 1000 nodes and 3000
    edges" regime while keeping the class-preference signal.
    """

    graph = adjacency.tolil().astype(np.int8)
    rng = np.random.default_rng(seed)
    partitions = [np.flatnonzero(labels == class_id) for class_id in range(int(labels.max()) + 1)]

    edge_pairs = {
        (int(i), int(j))
        for i, j in zip(*graph.nonzero(), strict=True)
        if int(i) < int(j)
    }

    if len(edge_pairs) > target_edges:
        removable = list(edge_pairs)
        rng.shuffle(removable)
        for source, target in removable[: len(edge_pairs) - target_edges]:
            graph[source, target] = 0
            graph[target, source] = 0
        graph = graph.todok()
        graph._shape = adjacency.shape
        return graph

    max_attempts = max(10000, 20 * target_edges)
    attempts = 0
    while len(edge_pairs) < target_edges and attempts < max_attempts:
        source = int(rng.integers(0, graph.shape[0]))
        source_class = int(labels[source])
        target_class = int(rng.choice(np.arange(preference_matrix.shape[0]), p=preference_matrix[source_class]))
        candidate_nodes = partitions[target_class]
        if candidate_nodes.size == 0:
            attempts += 1
            continue

        target = int(candidate_nodes[int(rng.integers(0, candidate_nodes.size))])
        pair = (source, target) if source < target else (target, source)
        if source == target or pair in edge_pairs:
            attempts += 1
            continue

        graph[source, target] = 1
        graph[target, source] = 1
        edge_pairs.add(pair)
        attempts += 1

    graph = graph.todok()
    graph._shape = adjacency.shape
    return graph


def _cache_key(dataset_config: dict[str, Any]) -> str:
    cache_payload = {"_loader_version": 3}
    cache_payload.update(
        {
            key: dataset_config[key]
            for key in sorted(dataset_config)
            if key not in {"loader", "root", "name"}
        }
    )
    raw = json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _build_synthetic_graph(dataset_config: dict[str, Any]) -> tuple[SyntheticGraphDataset, Data, dict[str, Any]]:
    gencat_module = _resolve_gencat_import()

    num_nodes = int(dataset_config.get("num_nodes", 1000))
    num_edges = int(dataset_config.get("num_edges", 3000))
    num_classes = int(dataset_config.get("num_classes", 6))
    num_features = int(dataset_config.get("num_features", 128))
    generation_seed = int(dataset_config.get("generation_seed", 2026))
    max_degree = int(dataset_config.get("max_degree", 60))
    phi_c = float(dataset_config.get("phi_c", 1.4))
    omega = float(dataset_config.get("omega", 0.12))
    r = int(dataset_config.get("r", 50))
    step = int(dataset_config.get("step", 100))
    att_type = str(dataset_config.get("att_type", "normal"))

    topology = dataset_config.get("topology", {})
    attributes = dataset_config.get("attributes", {})

    diagonal_preference = float(topology.get("diagonal_preference", 0.62))
    neighbor_preference = float(topology.get("neighbor_preference", 0.12))
    preference_deviation = float(topology.get("deviation", 0.04))

    primary_strength = float(attributes.get("primary_strength", 0.80))
    secondary_strength = float(attributes.get("secondary_strength", 0.15))
    background_strength = float(attributes.get("background_strength", 0.05))

    np.random.seed(generation_seed)
    torch.manual_seed(generation_seed)
    import random

    random.seed(generation_seed)

    M = _build_ring_preference_matrix(num_classes, diagonal_preference, neighbor_preference)
    D = _build_preference_deviation_matrix(num_classes, preference_deviation)
    H = _build_attribute_correlation_matrix(
        num_features,
        num_classes,
        primary_strength,
        secondary_strength,
        background_strength,
    )

    adjacency, features, labels = gencat_module.gencat(
        n=num_nodes,
        m=num_edges,
        k=num_classes,
        d=num_features,
        max_deg=max_degree,
        M=M,
        D=D,
        H=H,
        phi_c=phi_c,
        omega=omega,
        r=r,
        step=step,
        att_type=att_type,
    )
    adjacency = _match_target_edge_count(
        adjacency=adjacency,
        labels=np.asarray(labels),
        target_edges=num_edges,
        preference_matrix=M,
        seed=generation_seed,
    )

    edge_index, _ = from_scipy_sparse_matrix(adjacency.tocsr())
    data = Data(
        x=torch.as_tensor(np.asarray(features), dtype=torch.float32),
        edge_index=edge_index,
        y=torch.as_tensor(np.asarray(labels), dtype=torch.long),
    )
    dataset = SyntheticGraphDataset(
        name=str(dataset_config.get("name", "gencat_real_scenario")),
        num_features=num_features,
        num_classes=num_classes,
    )
    metadata = {
        "generation_seed": generation_seed,
        "num_nodes_requested": num_nodes,
        "num_edges_requested": num_edges,
        "num_classes_requested": num_classes,
        "num_features_requested": num_features,
        "max_degree": max_degree,
        "phi_c": phi_c,
        "omega": omega,
        "r": r,
        "step": step,
        "att_type": att_type,
        "topology": {
            "diagonal_preference": diagonal_preference,
            "neighbor_preference": neighbor_preference,
            "deviation": preference_deviation,
        },
        "attributes": {
            "primary_strength": primary_strength,
            "secondary_strength": secondary_strength,
            "background_strength": background_strength,
        },
    }
    return dataset, data, metadata


@DATASET_REGISTRY.register("gencat_synthetic")
def load_gencat_synthetic_from_config(dataset_config: dict[str, Any]) -> dict[str, Any]:
    dataset_root = resolve_project_path(dataset_config.get("root", "data/generated"))
    ensure_directory(dataset_root)

    cache_key = _cache_key(dataset_config)
    dataset_name = str(dataset_config.get("name", "gencat_real_scenario"))
    cache_path = dataset_root / f"{dataset_name}_{cache_key}.pt"

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        data = Data(
            x=payload["x"],
            edge_index=payload["edge_index"],
            y=payload["y"],
        )
        dataset = SyntheticGraphDataset(
            name=dataset_name,
            num_features=int(payload["num_features"]),
            num_classes=int(payload["num_classes"]),
        )
        generation_metadata = payload["generation_metadata"]
    else:
        dataset, data, generation_metadata = _build_synthetic_graph(dataset_config)
        torch.save(
            {
                "x": data.x.detach().cpu(),
                "edge_index": data.edge_index.detach().cpu(),
                "y": data.y.detach().cpu(),
                "num_features": int(dataset.num_features),
                "num_classes": int(dataset.num_classes),
                "generation_metadata": generation_metadata,
            },
            cache_path,
        )

    metadata = {
        "loader": "gencat_synthetic",
        "name": dataset_name,
        "root": str(dataset_root),
        "cache_path": str(cache_path),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges // 2 if data.num_edges % 2 == 0 else data.num_edges),
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
    }
    metadata.update(generation_metadata)

    return {
        "dataset": dataset,
        "data": data,
        "metadata": metadata,
    }
