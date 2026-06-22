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

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


DEFAULT_GAMMA_VALUES = [0.0, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0]
DEFAULT_PROBABILITY_VALUES = [round(value, 1) for value in np.arange(0.1, 1.0, 0.1)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DensGNN ablations for clustering_loss_gamma, warmup_epochs and point_probability_threshold.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "cora_benchmarks" / "cora_densgnn_core.yaml"),
        help="Base YAML config for DensGNN.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "densgnn_ablation"),
        help="Directory used to store summaries, histories and plots.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for the number of epochs used in every ablation run.",
    )
    parser.add_argument(
        "--gamma-values",
        nargs="+",
        type=float,
        default=DEFAULT_GAMMA_VALUES,
        help="Values used in the clustering_loss_gamma sweep.",
    )
    parser.add_argument(
        "--warmup-start",
        type=int,
        default=0,
        help="Initial warmup value for the warmup sweep.",
    )
    parser.add_argument(
        "--warmup-end",
        type=int,
        default=100,
        help="Final warmup value for the warmup sweep.",
    )
    parser.add_argument(
        "--warmup-step",
        type=int,
        default=10,
        help="Step size for the warmup sweep.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional override for the DensGNN device.",
    )
    parser.add_argument(
        "--probability-values",
        nargs="+",
        type=float,
        default=DEFAULT_PROBABILITY_VALUES,
        help="Values used in the point_probability_threshold sweep.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose metrics.json already exists.",
    )
    return parser


def cleanup_runtime_memory() -> None:
    gc.collect()

    if torch is None or not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def build_densgnn_config(
    dataset_bundle: dict[str, Any],
    base_config: dict[str, Any],
    *,
    gamma: float,
    warmup_epochs: int,
    point_probability_threshold: float | None,
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
        warmup_epochs=int(warmup_epochs),
        learning_rate=float(params.get("learning_rate", 0.01)),
        weight_decay=float(params.get("weight_decay", 0.0)),
        dropout=float(params.get("dropout", 0.0)),
        clustering_loss_gamma=float(gamma),
        update_p_interval=int(params.get("update_p_interval", 5)),
        point_selection=str(params.get("point_selection", "core")),
        point_probability_threshold=float(
            point_probability_threshold
            if point_probability_threshold is not None
            else params.get("point_probability_threshold", 0.5)
        ),
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


def plot_loss_grid(
    histories: dict[str, list[dict[str, float]]],
    output_path: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    metric_specs = [
        ("reconstruction_loss", "Reconstruction loss"),
        ("clustering_loss", "Clustering loss"),
        ("total_loss", "Total loss"),
    ]

    for axis, (metric_name, axis_title) in zip(axes, metric_specs, strict=True):
        for label, history in histories.items():
            epochs = [entry["epoch"] for entry in history]
            values = [entry[metric_name] for entry in history]
            axis.plot(epochs, values, label=label, linewidth=1.8)
        axis.set_title(axis_title)
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Epoch")
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_metric_curve(
    x_values: list[float | int],
    y_values: list[float],
    output_path: Path,
    *,
    x_label: str,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(x_values, y_values, marker="o", linewidth=2.0)
    axis.set_xlabel(x_label)
    axis.set_ylabel("NMI")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_summary_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_single_ablation(
    dataset_bundle: dict[str, Any],
    base_config: dict[str, Any],
    run_output_dir: Path,
    *,
    gamma: float,
    warmup_epochs: int,
    point_probability_threshold: float | None,
    epochs_override: int | None,
    device_override: str | None,
) -> dict[str, Any]:
    data = dataset_bundle["data"]
    labels = data.y.detach().cpu().numpy() if getattr(data, "y", None) is not None else None
    seed = int(base_config.get("run", {}).get("seed", 42))
    set_random_seed(seed)

    densgnn_config = build_densgnn_config(
        dataset_bundle,
        base_config,
        gamma=gamma,
        warmup_epochs=warmup_epochs,
        point_probability_threshold=point_probability_threshold,
        epochs_override=epochs_override,
        device_override=device_override,
    )
    output = train_densgnn(data=data, labels=labels, config=densgnn_config)

    metrics: dict[str, Any] = {
        "modularity": float(output["extra_metrics"].get("modularity", 0.0)),
        "num_clusters_found": int(np.unique(output["assignments"]).size),
    }
    if labels is not None:
        metrics.update(
            evaluate_clustering(
                features=output["embeddings"],
                labels=labels,
                assignments=output["assignments"],
            )
        )
    metrics.update(output["extra_metrics"])

    ensure_directory(run_output_dir)
    payload = {
        "gamma": gamma,
        "warmup_epochs": warmup_epochs,
        "point_probability_threshold": densgnn_config.point_probability_threshold,
        "seed": seed,
        "metrics": metrics,
        "loss_history": output["loss_history"],
        "evaluation_history": output["evaluation_history"],
    }
    save_json(run_output_dir / "summary.json", payload)

    return payload


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

    dataset_loader = DATASET_REGISTRY.get(dataset_loader_name)
    dataset_bundle = dataset_loader(dataset_config)

    gamma_rows: list[dict[str, Any]] = []
    gamma_histories: dict[str, list[dict[str, float]]] = {}
    for gamma in args.gamma_values:
        run_dir = output_dir / "gamma_sweep" / f"gamma_{gamma:g}"
        summary_path = run_dir / "summary.json"

        if args.skip_existing and summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            payload = run_single_ablation(
                dataset_bundle,
                base_config,
                run_dir,
                gamma=float(gamma),
                warmup_epochs=int(base_config.get("algorithm", {}).get("params", {}).get("warmup_epochs", 10)),
                point_probability_threshold=None,
                epochs_override=args.epochs,
                device_override=args.device,
            )
        gamma_histories[f"gamma={gamma:g}"] = payload["loss_history"]
        gamma_rows.append(
            {
                "gamma": float(gamma),
                "warmup_epochs": int(payload["warmup_epochs"]),
                "nmi": payload["metrics"].get("nmi"),
                "ari": payload["metrics"].get("ari"),
                "silhouette": payload["metrics"].get("silhouette"),
                "modularity": payload["metrics"].get("modularity"),
                "num_clusters_found": payload["metrics"].get("num_clusters_found"),
                "reconstruction_loss": payload["metrics"].get("reconstruction_loss"),
                "clustering_loss": payload["metrics"].get("clustering_loss"),
                "total_loss": payload["metrics"].get("total_loss"),
            }
        )
        cleanup_runtime_memory()

    save_json(output_dir / "gamma_sweep_summary.json", {"runs": gamma_rows})
    write_summary_csv(output_dir / "gamma_sweep_summary.csv", gamma_rows)
    plot_loss_grid(
        gamma_histories,
        output_dir / "gamma_sweep_loss.png",
        "DensGNN gamma sweep loss curves",
    )
    plot_metric_curve(
        [row["gamma"] for row in gamma_rows],
        [float(row["nmi"]) for row in gamma_rows if isinstance(row["nmi"], (int, float))],
        output_dir / "gamma_sweep_nmi.png",
        x_label="clustering_loss_gamma",
        title="DensGNN NMI across clustering_loss_gamma values",
    )

    warmup_values = list(range(args.warmup_start, args.warmup_end + 1, args.warmup_step))
    warmup_rows: list[dict[str, Any]] = []
    warmup_histories: dict[str, list[dict[str, float]]] = {}
    for warmup_epochs in warmup_values:
        run_dir = output_dir / "warmup_sweep" / f"warmup_{warmup_epochs}"
        summary_path = run_dir / "summary.json"

        if args.skip_existing and summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            payload = run_single_ablation(
                dataset_bundle,
                base_config,
                run_dir,
                gamma=float(base_config.get("algorithm", {}).get("params", {}).get("clustering_loss_gamma", 1.0)),
                warmup_epochs=warmup_epochs,
                point_probability_threshold=None,
                epochs_override=args.epochs,
                device_override=args.device,
            )
        warmup_histories[f"warmup={warmup_epochs}"] = payload["loss_history"]
        warmup_rows.append(
            {
                "gamma": float(payload["gamma"]),
                "warmup_epochs": int(warmup_epochs),
                "nmi": payload["metrics"].get("nmi"),
                "ari": payload["metrics"].get("ari"),
                "silhouette": payload["metrics"].get("silhouette"),
                "modularity": payload["metrics"].get("modularity"),
                "num_clusters_found": payload["metrics"].get("num_clusters_found"),
                "reconstruction_loss": payload["metrics"].get("reconstruction_loss"),
                "clustering_loss": payload["metrics"].get("clustering_loss"),
                "total_loss": payload["metrics"].get("total_loss"),
            }
        )
        cleanup_runtime_memory()

    save_json(output_dir / "warmup_sweep_summary.json", {"runs": warmup_rows})
    write_summary_csv(output_dir / "warmup_sweep_summary.csv", warmup_rows)
    plot_loss_grid(
        warmup_histories,
        output_dir / "warmup_sweep_loss.png",
        "DensGNN warmup sweep loss curves",
    )
    plot_metric_curve(
        [row["warmup_epochs"] for row in warmup_rows],
        [float(row["nmi"]) for row in warmup_rows if isinstance(row["nmi"], (int, float))],
        output_dir / "warmup_sweep_nmi.png",
        x_label="warmup_epochs",
        title="DensGNN NMI across warmup values",
    )

    probability_rows: list[dict[str, Any]] = []
    probability_histories: dict[str, list[dict[str, float]]] = {}
    for threshold in args.probability_values:
        run_dir = output_dir / "probability_sweep" / f"threshold_{threshold:.1f}"
        summary_path = run_dir / "summary.json"

        if args.skip_existing and summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            payload = run_single_ablation(
                dataset_bundle,
                base_config,
                run_dir,
                gamma=float(base_config.get("algorithm", {}).get("params", {}).get("clustering_loss_gamma", 1.0)),
                warmup_epochs=int(base_config.get("algorithm", {}).get("params", {}).get("warmup_epochs", 10)),
                point_probability_threshold=float(threshold),
                epochs_override=args.epochs,
                device_override=args.device,
            )
        probability_histories[f"threshold={threshold:.1f}"] = payload["loss_history"]
        probability_rows.append(
            {
                "gamma": float(payload["gamma"]),
                "warmup_epochs": int(payload["warmup_epochs"]),
                "point_probability_threshold": float(payload["point_probability_threshold"]),
                "nmi": payload["metrics"].get("nmi"),
                "ari": payload["metrics"].get("ari"),
                "silhouette": payload["metrics"].get("silhouette"),
                "modularity": payload["metrics"].get("modularity"),
                "num_clusters_found": payload["metrics"].get("num_clusters_found"),
                "reconstruction_loss": payload["metrics"].get("reconstruction_loss"),
                "clustering_loss": payload["metrics"].get("clustering_loss"),
                "total_loss": payload["metrics"].get("total_loss"),
            }
        )
        cleanup_runtime_memory()

    save_json(output_dir / "probability_sweep_summary.json", {"runs": probability_rows})
    write_summary_csv(output_dir / "probability_sweep_summary.csv", probability_rows)
    plot_loss_grid(
        probability_histories,
        output_dir / "probability_sweep_loss.png",
        "DensGNN point probability threshold sweep loss curves",
    )
    plot_metric_curve(
        [row["point_probability_threshold"] for row in probability_rows],
        [float(row["nmi"]) for row in probability_rows if isinstance(row["nmi"], (int, float))],
        output_dir / "probability_sweep_nmi.png",
        x_label="point_probability_threshold",
        title="DensGNN NMI across point probability thresholds",
    )

    print(f"DensGNN ablation finished. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
