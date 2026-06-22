from __future__ import annotations

from typing import Any

from graph_benchmark.register import register_all
from graph_benchmark.registry import EXPERIMENT_REGISTRY


def run_from_config(config: dict[str, Any]) -> dict[str, Any]:
    register_all()

    run_config = config.get("run", {})
    experiment_name = run_config.get("experiment")

    if not experiment_name:
        raise KeyError("Missing 'run.experiment' in the YAML config.")

    experiment = EXPERIMENT_REGISTRY.get(experiment_name)
    return experiment(config)
