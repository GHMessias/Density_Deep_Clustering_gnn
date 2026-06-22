from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_benchmark.config.yaml import load_yaml_config  # noqa: E402
from graph_benchmark.evaluation.clustering import evaluate_clustering  # noqa: E402
from graph_benchmark.models.densgnn import DensGNNConfig, train_densgnn  # noqa: E402
from graph_benchmark.register import register_all  # noqa: E402
from graph_benchmark.registry import DATASET_REGISTRY  # noqa: E402
from graph_benchmark.utils.io import ensure_directory, save_json  # noqa: E402
from graph_benchmark.utils.seed import set_random_seed  # noqa: E402


# Ten variance values grouped into three regimes.
# We use actual variances instead of a generic "low/medium/high" label so the
# experiment is easy to reproduce and discuss in the paper.
DEFAULT_VARIANCE_VALUES = [0.0004, 0.0009, 0.0016, 0.0025, 0.0049, 0.0100, 0.0225, 0.0400, 0.0900, 0.1600]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a DensGNN variance study with fixed topology and GenCAT-style "
            "continuous attributes generated from class prototypes plus Gaussian noise."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "cora_benchmarks" / "cora_densgnn_core.yaml"),
        help="Base DensGNN YAML config. The graph topology and model settings are loaded from here.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "densgnn_gencat_variance_curve"),
        help="Directory used to save summaries and plots.",
    )
    parser.add_argument(
        "--variance-values",
        nargs="+",
        type=float,
        default=DEFAULT_VARIANCE_VALUES,
        help="Noise variances used to generate the continuous attributes.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for the number of DensGNN epochs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional override for the DensGNN device.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip individual runs whose summary.json already exists.",
    )
    return parser


def cleanup_runtime_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def build_densgnn_config(
    dataset_bundle: dict[str, Any],
    base_config: dict[str, Any],
    *,
    epochs_override: int | None,
    device_override: str | None,
) -> DensGNNConfig:
    params = base_config.get("algorithm", {}).get("params", {})
    dataset = dataset_bundle["dataset"]

    return DensGNNConfig(
        input_channels=int(dataset.num_features),
        hidden_channels=int(params.get("encoder_hidden_channels", 128)),
        embedding_channels=int(params.get("embedding_channels", 32)),
        epochs=int(epochs_override if epochs_override is not None else params.get("epochs", 100)),
        warmup_epochs=int(params.get("warmup_epochs", 10)),
        learning_rate=float(params.get("learning_rate", 0.01)),
        weight_decay=float(params.get("weight_decay", 0.0)),
        dropout=float(params.get("dropout", 0.0)),
        clustering_loss_gamma=float(params.get("clustering_loss_gamma", 1.0)),
        update_p_interval=int(params.get("update_p_interval", 5)),
        point_selection=str(params.get("point_selection", "core")),
        point_probability_threshold=float(params.get("point_probability_threshold", 0.5)),
        hdbscan_min_cluster_size=int(params.get("hdbscan_min_cluster_size", 10)),
        hdbscan_min_samples=int(params.get("hdbscan_min_samples", 5)),
        hdbscan_cluster_selection_method=str(params.get("hdbscan_cluster_selection_method", "eom")),
        mrd_k=int(params.get("mrd_k", params.get("hdbscan_min_samples", 5))),
        random_state=int(base_config.get("run", {}).get("seed", 42)),
        device=str(device_override if device_override is not None else params.get("device", "auto")),
        verbose=bool(params.get("verbose", False)),
        log_interval=int(params.get("log_interval", 1)),
        evaluation_interval=int(params.get("evaluation_interval", 5)),
    )


def compute_class_attribute_prototypes(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute GenCAT-style class prototypes H from the original dataset.

    In the GenCAT workbench, H is the attribute-class correlation matrix: the
    average activation of each attribute inside each class. We reuse the same
    idea here, but keep the original graph topology fixed. This lets us vary
    only the dispersion of X while preserving class semantics.
    """

    num_classes = int(labels.max()) + 1
    num_features = int(features.shape[1])
    H = np.zeros((num_features, num_classes), dtype=np.float32)

    for class_id in range(num_classes):
        class_mask = labels == class_id
        H[:, class_id] = features[class_mask].mean(axis=0)

    return H


def build_continuous_features_from_variance(
    labels: np.ndarray,
    class_attribute_prototypes: np.ndarray,
    *,
    variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate continuous attributes by mirroring GenCAT's normal branch.

    Design choice:
    - Each node starts from the prototype of its class.
    - Gaussian noise with the requested variance is added feature-wise.
    - Each feature column is min-max normalized to [0, 1], matching the
      normalization step used in GenCAT/gencat.py for continuous attributes.

    This does *not* test sparsity directly. It tests whether DensGNN benefits
    from more compact or more dispersed attribute clouds while the topology
    stays fixed.
    """

    std = float(np.sqrt(variance))
    class_means = class_attribute_prototypes[:, labels].T
    noisy_features = class_means + rng.normal(loc=0.0, scale=std, size=class_means.shape).astype(np.float32)

    normalized = noisy_features.copy()
    for feature_idx in range(normalized.shape[1]):
        column = normalized[:, feature_idx]
        column_min = float(column.min())
        column_max = float(column.max())
        column_range = column_max - column_min
        if column_range <= 1e-12:
            normalized[:, feature_idx] = 0.0
            continue
        normalized[:, feature_idx] = (column - column_min) / column_range

    return normalized.astype(np.float32)


def estimate_feature_dispersion(features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return simple intra/inter-class distance summaries for interpretation."""

    num_classes = int(labels.max()) + 1
    class_centroids = np.zeros((num_classes, features.shape[1]), dtype=np.float32)
    intra_distances: list[float] = []

    for class_id in range(num_classes):
        class_mask = labels == class_id
        class_features = features[class_mask]
        centroid = class_features.mean(axis=0)
        class_centroids[class_id] = centroid
        intra_distances.append(float(np.mean(np.linalg.norm(class_features - centroid, axis=1))))

    inter_distances: list[float] = []
    for class_i in range(num_classes):
        for class_j in range(class_i + 1, num_classes):
            inter_distances.append(float(np.linalg.norm(class_centroids[class_i] - class_centroids[class_j])))

    return float(np.mean(intra_distances)), float(np.mean(inter_distances))


def variance_scenario(index: int, total: int) -> str:
    third = max(1, total // 3)
    if index < third:
        return "low"
    if index < 2 * third:
        return "medium"
    return "high"


def write_summary_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(
    x_values: list[float],
    y_values: list[float],
    output_path: Path,
    *,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(x_values, y_values, marker="o", linewidth=2.0)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    ensure_directory(output_dir)

    register_all()
    base_config = load_yaml_config(config_path)

    dataset_config = base_config.get("dataset", {})
    dataset_loader_name = dataset_config.get("loader")
    if not dataset_loader_name:
        raise KeyError("Missing 'dataset.loader' in the base config.")

    dataset_loader = DATASET_REGISTRY.get(str(dataset_loader_name))
    dataset_bundle = dataset_loader(dataset_config)
    data = dataset_bundle["data"]
    labels = data.y.detach().cpu().numpy()
    original_features = data.x.detach().cpu().numpy().astype(np.float32)
    class_attribute_prototypes = compute_class_attribute_prototypes(original_features, labels)

    base_seed = int(base_config.get("run", {}).get("seed", 42))
    densgnn_config = build_densgnn_config(
        dataset_bundle,
        base_config,
        epochs_override=args.epochs,
        device_override=args.device,
    )

    summary_rows: list[dict[str, Any]] = []
    summary_payloads: list[dict[str, Any]] = []

    for index, variance in enumerate(args.variance_values):
        scenario = variance_scenario(index, len(args.variance_values))
        run_dir = output_dir / "variance_sweep" / f"variance_{variance:.4f}"
        summary_path = run_dir / "summary.json"

        if args.skip_existing and summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_rows.append(payload["summary"])
            summary_payloads.append(payload)
            continue

        run_seed = base_seed + index
        set_random_seed(run_seed)
        rng = np.random.default_rng(run_seed)

        synthetic_features = build_continuous_features_from_variance(
            labels=labels,
            class_attribute_prototypes=class_attribute_prototypes,
            variance=float(variance),
            rng=rng,
        )

        intra_distance, inter_distance = estimate_feature_dispersion(synthetic_features, labels)

        synthetic_data = data.clone()
        synthetic_data.x = torch.from_numpy(synthetic_features)

        output = train_densgnn(data=synthetic_data, labels=labels, config=densgnn_config)
        metrics = evaluate_clustering(
            features=output["embeddings"],
            labels=labels,
            assignments=output["assignments"],
        )
        metrics.update(output["extra_metrics"])

        summary_row = {
            "variance": float(variance),
            "std": float(np.sqrt(variance)),
            "scenario": scenario,
            "seed": int(run_seed),
            "observed_feature_mean": float(synthetic_features.mean()),
            "observed_feature_variance": float(synthetic_features.var()),
            "mean_intra_class_distance": intra_distance,
            "mean_inter_class_distance": inter_distance,
            "nmi": metrics.get("nmi"),
            "ari": metrics.get("ari"),
            "clustering_accuracy": metrics.get("clustering_accuracy"),
            "silhouette": metrics.get("silhouette"),
            "modularity": metrics.get("modularity"),
            "num_clusters_found": metrics.get("num_clusters_found"),
            "reconstruction_loss": metrics.get("reconstruction_loss"),
            "clustering_loss": metrics.get("clustering_loss"),
            "total_loss": metrics.get("total_loss"),
        }
        payload = {
            "summary": summary_row,
            "notes": {
                "topology_policy": "The original dataset topology and labels are kept fixed.",
                "attribute_policy": (
                    "Synthetic continuous attributes are generated from class prototypes "
                    "plus Gaussian noise, following the normal branch of GenCAT."
                ),
                "variance_policy": (
                    "Only the Gaussian noise variance changes across runs. This isolates "
                    "the effect of attribute dispersion instead of Bernoulli sparsification."
                ),
            },
        }
        save_json(summary_path, payload)

        summary_rows.append(summary_row)
        summary_payloads.append(payload)
        cleanup_runtime_memory()

    save_json(
        output_dir / "variance_sweep_summary.json",
        {
            "config": {
                "base_config": str(config_path),
                "variance_values": [float(value) for value in args.variance_values],
            },
            "runs": summary_payloads,
        },
    )
    write_summary_csv(output_dir / "variance_sweep_summary.csv", summary_rows)

    sorted_rows = sorted(summary_rows, key=lambda row: float(row["variance"]))
    variance_values = [float(row["variance"]) for row in sorted_rows]
    nmis = [float(row["nmi"]) for row in sorted_rows]
    aris = [float(row["ari"]) for row in sorted_rows]
    accuracies = [float(row["clustering_accuracy"]) for row in sorted_rows]
    intra_distances = [float(row["mean_intra_class_distance"]) for row in sorted_rows]
    inter_distances = [float(row["mean_inter_class_distance"]) for row in sorted_rows]

    plot_curve(
        variance_values,
        nmis,
        output_dir / "nmi_vs_variance.png",
        x_label="Noise variance",
        y_label="NMI",
        title="DensGNN NMI across continuous attribute variance levels",
    )
    plot_curve(
        variance_values,
        aris,
        output_dir / "ari_vs_variance.png",
        x_label="Noise variance",
        y_label="ARI",
        title="DensGNN ARI across continuous attribute variance levels",
    )
    plot_curve(
        variance_values,
        accuracies,
        output_dir / "accuracy_vs_variance.png",
        x_label="Noise variance",
        y_label="Clustering accuracy",
        title="DensGNN clustering accuracy across continuous attribute variance levels",
    )
    plot_curve(
        variance_values,
        intra_distances,
        output_dir / "intra_distance_vs_variance.png",
        x_label="Noise variance",
        y_label="Mean intra-class distance",
        title="Attribute intra-class dispersion across variance levels",
    )
    plot_curve(
        variance_values,
        inter_distances,
        output_dir / "inter_distance_vs_variance.png",
        x_label="Noise variance",
        y_label="Mean inter-class centroid distance",
        title="Attribute inter-class separation across variance levels",
    )

    print(f"DensGNN variance experiment finished. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
