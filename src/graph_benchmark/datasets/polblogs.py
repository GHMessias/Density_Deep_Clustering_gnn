from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import PolBlogs

from graph_benchmark.registry import DATASET_REGISTRY
from graph_benchmark.utils.io import resolve_project_path


def load_polblogs_dataset(root: str | Path) -> tuple[PolBlogs, Data]:
    dataset_root = Path(root).expanduser()
    dataset = PolBlogs(root=str(dataset_root))
    data = dataset[0]

    if getattr(data, "x", None) is None:
        # PolBlogs ships without node attributes, so we use the identity matrix
        # as a simple feature encoding that preserves node individuality.
        data.x = torch.eye(data.num_nodes, dtype=torch.float32)

    return dataset, data


@DATASET_REGISTRY.register("polblogs")
def load_polblogs_from_config(dataset_config: dict[str, Any]) -> dict[str, Any]:
    dataset_root = resolve_project_path(dataset_config.get("root", "data/raw"))
    dataset_name = str(dataset_config.get("name", "PolBlogs"))
    dataset, data = load_polblogs_dataset(dataset_root)

    return {
        "dataset": dataset,
        "data": data,
        "metadata": {
            "loader": "polblogs",
            "name": dataset_name,
            "root": str(dataset_root),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
            "num_features": int(dataset.num_features if dataset.num_features > 0 else data.x.size(-1)),
            "num_classes": int(dataset.num_classes),
        },
    }
