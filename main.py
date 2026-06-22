from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_benchmark.config.yaml import apply_overrides, load_yaml_config  # noqa: E402
from graph_benchmark.runner import run_from_config  # noqa: E402
from graph_benchmark.utils.seed import set_random_seed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run graph benchmark experiments from YAML configs.",
    )
    parser.add_argument(
        "--cfg",
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Optional dotted overrides such as algorithm.params.n_init=50.",
    )
    return parser


def print_summary(results: dict[str, object]) -> None:
    run_info = results.get("run", {})
    dataset_info = results.get("dataset", {})
    algorithm_info = results.get("algorithm", {})
    metrics = results.get("metrics", {})
    artifacts = results.get("artifacts", {})

    if run_info:
        print(f"Experiment: {run_info.get('experiment')}")
    if dataset_info:
        print(f"Dataset: {dataset_info.get('name')}")
    if algorithm_info:
        print(f"Algorithm: {algorithm_info.get('name')}")

    for metric_name in (
        "nmi",
        "ari",
        "purity",
        "clustering_accuracy",
        "modularity",
        "silhouette",
        "reconstruction_loss",
        "clustering_loss",
        "total_loss",
        "best_eval_nmi",
        "best_eval_epoch",
        "best_eval_num_clusters",
        "inertia",
        "hdbscan_num_clusters",
        "hdbscan_noise_ratio",
        "support_points",
        "ricci_num_connected_components",
        "ricci_mean_cutoff",
    ):
        metric_value = metrics.get(metric_name)
        if isinstance(metric_value, (int, float)):
            print(f"{metric_name.upper()}: {metric_value:.4f}")

    output_dir = artifacts.get("output_dir")
    if output_dir:
        print(f"Results saved to: {output_dir}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_yaml_config(args.cfg)

    if args.set:
        config = apply_overrides(config, args.set)

    raw_seed = config.get("run", {}).get("seed", 42)
    run_seed = None if raw_seed is None else int(raw_seed)
    set_random_seed(run_seed)
    results = run_from_config(config)
    print_summary(results)


if __name__ == "__main__":
    main()
