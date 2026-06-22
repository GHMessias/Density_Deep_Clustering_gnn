from __future__ import annotations

from pathlib import Path
from typing import Any

from torch_geometric.data import Data
from torch_geometric.datasets import Amazon

from graph_benchmark.registry import DATASET_REGISTRY
from graph_benchmark.utils.io import resolve_project_path


def load_amazon_dataset(name: str, root: str | Path) -> tuple[Amazon, Data]:
    dataset_root = Path(root).expanduser()
    dataset = Amazon(root=str(dataset_root), name=name)
    return dataset, dataset[0]


@DATASET_REGISTRY.register("amazon")
def load_amazon_from_config(dataset_config: dict[str, Any]) -> dict[str, Any]:
    dataset_name = dataset_config.get("name")
    if not dataset_name:
        raise KeyError("Missing 'dataset.name' for the amazon loader.")

    dataset_root = resolve_project_path(dataset_config.get("root", "data/raw"))
    dataset, data = load_amazon_dataset(dataset_name, dataset_root)

    return {
        "dataset": dataset,
        "data": data,
        "metadata": {
            "loader": "amazon",
            "name": dataset_name,
            "root": str(dataset_root),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
            "num_features": int(dataset.num_features),
            "num_classes": int(dataset.num_classes),
        },
    }
