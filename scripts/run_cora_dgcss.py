from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_benchmark.config.yaml import apply_overrides, load_yaml_config  # noqa: E402
from graph_benchmark.runner import run_from_config  # noqa: E402
from graph_benchmark.utils.seed import set_random_seed  # noqa: E402


def default_config_path() -> Path:
    return PROJECT_ROOT / "configs" / "cora_dgcss_bc.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for the YAML-driven Cora + DGCSS benchmark.",
    )
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Override the dataset download directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory for metrics and assignments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the global seed for the benchmark run.",
    )
    parser.add_argument(
        "--seed-selector",
        default=None,
        help="Override the seed selection strategy, e.g. betweenness_centrality, closeness_centrality, random.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml_config(args.config)
    overrides = []

    if args.dataset_root is not None:
        overrides.append(f"dataset.root={args.dataset_root}")
    if args.output_dir is not None:
        overrides.append(f"output.dir={args.output_dir}")
    if args.seed is not None:
        overrides.append(f"run.seed={args.seed}")
        overrides.append(f"algorithm.params.random_state={args.seed}")
    if args.seed_selector is not None:
        overrides.append(f"algorithm.params.seed_selector={args.seed_selector}")

    if overrides:
        config = apply_overrides(config, overrides)

    run_seed = int(config.get("run", {}).get("seed", 42))
    set_random_seed(run_seed)
    results = run_from_config(config)
    metrics = results["metrics"]

    print(f"Experiment: {results['run']['experiment']}")
    print(f"Dataset: {results['dataset']['name']}")
    print(f"Algorithm: {results['algorithm']['name']}")
    print(f"NMI: {metrics['nmi']:.4f}")
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"Purity: {metrics['purity']:.4f}")
    print(f"Accuracy: {metrics['clustering_accuracy']:.4f}")
    print(f"Modularity: {metrics['modularity']:.4f}")
    print(f"Reconstruction loss: {metrics['reconstruction_loss']:.4f}")
    print(f"Clustering loss: {metrics['clustering_loss']:.4f}")
    print(f"Total loss: {metrics['total_loss']:.4f}")
    print(f"Results saved to: {results['artifacts']['output_dir']}")


if __name__ == "__main__":
    main()
