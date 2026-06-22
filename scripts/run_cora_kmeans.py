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

from graph_benchmark.experiments.cora_kmeans_features import (  # noqa: E402
    apply_overrides,
    default_config_path,
    load_experiment_config,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for the YAML-driven Cora + KMeans benchmark.",
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
        help="Override the random seed used by KMeans.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    config = apply_overrides(
        config,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        random_state=args.seed,
    )
    results = run_experiment(config)
    metrics = results["metrics"]

    print(f"Experiment: {results['run']['experiment']}")
    print(f"Dataset: {results['dataset']['name']}")
    print(f"Algorithm: {results['algorithm']['name']}")
    print(f"NMI: {metrics['nmi']:.4f}")
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"Purity: {metrics['purity']:.4f}")
    if "silhouette" in metrics:
        print(f"Silhouette: {metrics['silhouette']:.4f}")
    print(f"Inertia: {metrics['inertia']:.4f}")
    print(f"Results saved to: {results['artifacts']['output_dir']}")


if __name__ == "__main__":
    main()
