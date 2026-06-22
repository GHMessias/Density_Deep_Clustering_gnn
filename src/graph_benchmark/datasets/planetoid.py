from __future__ import annotations

from pathlib import Path
from typing import Any

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from graph_benchmark.registry import DATASET_REGISTRY
from graph_benchmark.utils.io import resolve_project_path


def load_planetoid_dataset(name: str, root: str | Path) -> tuple[Planetoid, Data]:
    dataset_root = Path(root).expanduser()
    dataset = Planetoid(root=str(dataset_root), name=name)
    return dataset, dataset[0]


@DATASET_REGISTRY.register("planetoid")
def load_planetoid_from_config(dataset_config: dict[str, Any]) -> dict[str, Any]:
    dataset_name = dataset_config.get("name")
    if not dataset_name:
        raise KeyError("Missing 'dataset.name' for the planetoid loader.")

    dataset_root = resolve_project_path(dataset_config.get("root", "data/raw"))
    dataset, data = load_planetoid_dataset(dataset_name, dataset_root)

    return {
        "dataset": dataset,
        "data": data,
        "metadata": {
            "loader": "planetoid",
            "name": dataset_name,
            "root": str(dataset_root),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
            "num_features": int(dataset.num_features),
            "num_classes": int(dataset.num_classes),
        },
    }
