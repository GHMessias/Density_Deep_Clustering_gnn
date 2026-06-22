from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Twitch

from graph_benchmark.registry import DATASET_REGISTRY
from graph_benchmark.utils.io import resolve_project_path


def load_twitch_dataset(name: str, root: str | Path) -> tuple[Twitch, Data]:
    dataset_root = Path(root).expanduser()
    try:
        dataset = Twitch(root=str(dataset_root), name=name)
    except Exception as exc:  # noqa: BLE001
        expected_raw_path = dataset_root / name / "raw" / f"{name}.npz"
        raise RuntimeError(
            "Failed to load the Twitch dataset split "
            f"{name!r}. The split name is valid for PyG, but the upstream file "
            "download failed. If the remote host is returning 404, place the "
            f"raw file manually at '{expected_raw_path}'."
        ) from exc
    data = dataset[0]

    if getattr(data, "x", None) is None:
        # Keep the loader robust if a Twitch split ever ships without explicit
        # node features in a local cache. The identity matrix is the same
        # fallback already used for PolBlogs in this benchmark suite.
        data.x = torch.eye(data.num_nodes, dtype=torch.float32)

    return dataset, data


@DATASET_REGISTRY.register("twitch")
def load_twitch_from_config(dataset_config: dict[str, Any]) -> dict[str, Any]:
    dataset_name = dataset_config.get("name")
    if not dataset_name:
        raise KeyError("Missing 'dataset.name' for the twitch loader.")

    dataset_root = resolve_project_path(dataset_config.get("root", "data/raw"))
    dataset, data = load_twitch_dataset(dataset_name, dataset_root)

    num_features = int(dataset.num_features if dataset.num_features > 0 else data.x.size(-1))

    return {
        "dataset": dataset,
        "data": data,
        "metadata": {
            "loader": "twitch",
            "name": dataset_name,
            "root": str(dataset_root),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
            "num_features": num_features,
            "num_classes": int(dataset.num_classes),
        },
    }
