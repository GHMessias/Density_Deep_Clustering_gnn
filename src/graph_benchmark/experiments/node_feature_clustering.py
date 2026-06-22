from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from graph_benchmark.evaluation.clustering import evaluate_clustering
from graph_benchmark.registry import ALGORITHM_REGISTRY, DATASET_REGISTRY, EXPERIMENT_REGISTRY
from graph_benchmark.utils.graph_ops import compute_modularity
from graph_benchmark.utils.io import ensure_directory, resolve_project_path, save_json
from graph_benchmark.utils.seed import set_random_seed


@EXPERIMENT_REGISTRY.register("node_feature_clustering")
def run_node_feature_clustering(config: dict[str, Any]) -> dict[str, Any]:
    run_config = config.get("run", {})
    dataset_config = config.get("dataset", {})
    algorithm_config = config.get("algorithm", {})
    output_config = config.get("output", {})

    raw_seed = run_config.get("seed", 42)
    seed = None if raw_seed is None else int(raw_seed)
    set_random_seed(seed)

    dataset_loader_name = dataset_config.get("loader")
    if not dataset_loader_name:
        raise KeyError("Missing 'dataset.loader' in the YAML config.")

    algorithm_name = algorithm_config.get("name")
    if not algorithm_name:
        raise KeyError("Missing 'algorithm.name' in the YAML config.")

    dataset_loader = DATASET_REGISTRY.get(dataset_loader_name)
    algorithm_runner = ALGORITHM_REGISTRY.get(algorithm_name)

    dataset_bundle = dataset_loader(dataset_config)
    algorithm_config = deepcopy(algorithm_config)
    algorithm_config.setdefault("params", {})
    if seed is not None:
        algorithm_config["params"].setdefault("random_state", seed)
    algorithm_output = algorithm_runner(dataset_bundle, algorithm_config)

    data = dataset_bundle["data"]
    labels = data.y.detach().cpu().numpy() if getattr(data, "y", None) is not None else None
    assignments = np.asarray(algorithm_output["assignments"])
    metrics: dict[str, Any] = {
        "nmi": None,
        "ari": None,
        "modularity": float(compute_modularity(data, assignments)),
        "num_clusters_found": int(len(np.unique(assignments))),
    }

    if labels is not None:
        metrics.update(
            evaluate_clustering(
            features=algorithm_output.get("features"),
            labels=labels,
            assignments=assignments,
        )
        )

    if algorithm_output.get("inertia") is not None:
        metrics["inertia"] = float(algorithm_output["inertia"])
    metrics.update(algorithm_output.get("extra_metrics", {}))

    output_dir = resolve_project_path(
        output_config.get(
            "dir",
            f"results/{dataset_bundle['metadata']['name'].lower()}/{algorithm_name}",
        )
    )
    ensure_directory(output_dir)

    results = {
        "run": {
            "experiment": run_config.get("experiment"),
            "seed": seed,
        },
        "dataset": dataset_bundle["metadata"],
        "algorithm": algorithm_output["metadata"],
        "metrics": metrics,
        "artifacts": {
            "output_dir": str(output_dir),
        },
        "config": config,
    }

    metrics_path = output_dir / "metrics.json"
    results["artifacts"]["metrics_path"] = str(metrics_path)

    if output_config.get("save_assignments", True):
        assignments_path = output_dir / "assignments.csv"
        labels_column = labels if labels is not None else np.full(data.num_nodes, -1, dtype=int)
        rows = np.column_stack((np.arange(data.num_nodes), labels_column, assignments))
        np.savetxt(
            assignments_path,
            rows,
            delimiter=",",
            fmt="%d",
            header="node_id,label,cluster",
            comments="",
        )
        results["artifacts"]["assignments_path"] = str(assignments_path)

    if output_config.get("save_features", True):
        features = algorithm_output.get("features")
        if features is not None:
            features_path = output_dir / "features.npy"
            np.save(features_path, np.asarray(features))
            results["artifacts"]["features_path"] = str(features_path)

    save_json(metrics_path, results)

    return results
