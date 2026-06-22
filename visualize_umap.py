from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import umap
import sys

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_benchmark.utils.seed import set_random_seed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a UMAP visualization from a benchmark result directory.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing metrics.json, assignments.csv and features.npy.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Defaults to <results-dir>/umap_projection.png",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="UMAP metric parameter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random state used by UMAP.",
    )
    return parser


def load_assignments(assignments_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(assignments_path, delimiter=",", skiprows=1, dtype=int)
    true_labels = rows[:, 1]
    predicted_labels = rows[:, 2]
    return true_labels, predicted_labels


def scatter_labels(ax, projection: np.ndarray, labels: np.ndarray, title: str) -> None:
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20")

    for label in unique_labels:
        mask = labels == label
        if label == -1:
            color = "#9a9a9a"
            plot_label = "noise"
        else:
            color = cmap(int(label) % 20)
            plot_label = str(int(label))
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=12,
            alpha=0.85,
            c=[color],
            label=plot_label,
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    args = build_parser().parse_args()
    set_random_seed(args.seed)
    results_dir = Path(args.results_dir).expanduser().resolve()
    features_path = results_dir / "features.npy"
    assignments_path = results_dir / "assignments.csv"

    if not features_path.exists():
        raise SystemExit(
            f"features.npy not found in {results_dir}. "
            "Re-run the experiment with output.save_features=true."
        )
    if not assignments_path.exists():
        raise SystemExit(
            f"assignments.csv not found in {results_dir}. "
            "Re-run the experiment with output.save_assignments=true."
        )

    features = np.load(features_path)
    true_labels, predicted_labels = load_assignments(assignments_path)

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    projection = reducer.fit_transform(features)

    has_true_labels = np.any(true_labels >= 0)
    if has_true_labels:
        figure, axes = plt.subplots(1, 2, figsize=(14, 6))
        scatter_labels(axes[0], projection, true_labels, "True Labels")
        scatter_labels(axes[1], projection, predicted_labels, "Predicted Clusters")
    else:
        figure, axes = plt.subplots(1, 1, figsize=(7, 6))
        axes = np.asarray([axes])
        scatter_labels(axes[0], projection, predicted_labels, "Predicted Clusters")

    figure.tight_layout()
    output_path = Path(args.output).expanduser().resolve() if args.output else results_dir / "umap_projection.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(f"Results directory: {results_dir}")
    print(f"Features path: {features_path}")
    print(f"Assignments path: {assignments_path}")
    print(f"UMAP projection saved to: {output_path}")


if __name__ == "__main__":
    main()
