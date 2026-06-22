from __future__ import annotations

from pathlib import Path
from typing import Any

from graph_benchmark.config.yaml import apply_overrides as apply_yaml_overrides
from graph_benchmark.config.yaml import load_yaml_config
from graph_benchmark.runner import run_from_config
from graph_benchmark.utils.io import project_root
from graph_benchmark.utils.seed import set_random_seed


def default_config_path() -> Path:
    return project_root() / "configs" / "cora_kmeans_features.yaml"


def load_experiment_config(config_path: str | Path | None = None) -> dict[str, Any]:
    resolved_config_path = Path(config_path) if config_path else default_config_path()
    return load_yaml_config(resolved_config_path)


def apply_overrides(
    config: dict[str, Any],
    dataset_root: str | None = None,
    output_dir: str | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    overrides = []

    if dataset_root is not None:
        overrides.append(f"dataset.root={dataset_root}")

    if output_dir is not None:
        overrides.append(f"output.dir={output_dir}")

    if random_state is not None:
        overrides.append(f"run.seed={random_state}")
        overrides.append(f"algorithm.params.random_state={random_state}")

    return apply_yaml_overrides(config, overrides) if overrides else config


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    run_seed = int(config.get("run", {}).get("seed", 42))
    set_random_seed(run_seed)
    return run_from_config(config)
