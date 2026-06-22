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


DEFAULT_P_VALUES = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a DensGNN density experiment with fixed Cora-like topology and "
            "GenCAT-style class-conditioned Bernoulli attributes."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "cora_benchmarks" / "cora_densgnn_core.yaml"),
        help="Base DensGNN YAML config. The dataset topology and model parameters are loaded from here.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "densgnn_gencat_density_curve"),
        help="Directory used to save summaries and plots.",
    )
    parser.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=DEFAULT_P_VALUES,
        help=(
            "Global densification factors. Each value rescales the Bernoulli "
            "probabilities before sampling the synthetic attribute matrix."
        ),
    )
    parser.add_argument(
        "--global-mix",
        type=float,
        default=0.25,
        help=(
            "Mixture weight for a dataset-wide attribute prior. This avoids a "
            "purely class-block sparse regime and makes the sparsity shift more global."
        ),
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


def compute_gen_cat_style_attribute_correlation(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return the GenCAT-style attribute-class correlation matrix H.

    This mirrors the logic of ``GenCAT/GenCAT_Workbench/func.py::calc_attr_cor``:
    for each class, compute the mean activation of every attribute among nodes
    that belong to that class. We reimplement it locally to avoid depending on
    the workbench module import path and its extra notebook-only dependencies.
    """

    num_classes = int(labels.max()) + 1
    num_features = int(features.shape[1])
    H = np.zeros((num_features, num_classes), dtype=np.float32)

    for class_id in range(num_classes):
        class_mask = labels == class_id
        H[:, class_id] = features[class_mask].mean(axis=0)

    return H


def build_density_controlled_features(
    labels: np.ndarray,
    class_attribute_correlation: np.ndarray,
    global_feature_prior: np.ndarray,
    *,
    density_scale: float,
    global_mix: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a synthetic binary feature matrix with a global density knob.

    Design choice:
    - The class-conditioned term comes from GenCAT's H matrix.
    - We blend in a global feature prior so the experiment is not sparse only at
      the class level. This makes the density shift more global, which was the
      user's explicit goal.
    - The scalar ``density_scale`` multiplies the blended Bernoulli
      probabilities. Values below 1 sparsify the matrix; values above 1 densify
      it (with clipping at 1).
    """

    class_probs = class_attribute_correlation[:, labels].T
    base_probabilities = (1.0 - global_mix) * class_probs + global_mix * global_feature_prior[None, :]
    scaled_probabilities = np.clip(density_scale * base_probabilities, 0.0, 1.0).astype(np.float32)
    sampled_features = rng.binomial(1, scaled_probabilities).astype(np.float32)
    return sampled_features, scaled_probabilities


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

    # We keep the topology fixed and only regenerate X. This isolates the
    # effect of attribute density on DensGNN, which was the target question.
    H = compute_gen_cat_style_attribute_correlation(original_features, labels)
    global_feature_prior = original_features.mean(axis=0).astype(np.float32)

    base_seed = int(base_config.get("run", {}).get("seed", 42))
    densgnn_config = build_densgnn_config(
        dataset_bundle,
        base_config,
        epochs_override=args.epochs,
        device_override=args.device,
    )

    summary_rows: list[dict[str, Any]] = []
    summary_payloads: list[dict[str, Any]] = []

    for index, density_scale in enumerate(args.p_values):
        run_dir = output_dir / "density_sweep" / f"p_{density_scale:g}"
        summary_path = run_dir / "summary.json"

        if args.skip_existing and summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_rows.append(payload["summary"])
            summary_payloads.append(payload)
            continue

        run_seed = base_seed + index
        set_random_seed(run_seed)
        rng = np.random.default_rng(run_seed)

        synthetic_features, feature_probabilities = build_density_controlled_features(
            labels=labels,
            class_attribute_correlation=H,
            global_feature_prior=global_feature_prior,
            density_scale=float(density_scale),
            global_mix=float(args.global_mix),
            rng=rng,
        )

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
            "p_scale": float(density_scale),
            "seed": int(run_seed),
            "global_mix": float(args.global_mix),
            "observed_feature_density": float(synthetic_features.mean()),
            "mean_sampling_probability": float(feature_probabilities.mean()),
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
                    "Synthetic X is sampled from Bernoulli probabilities derived from "
                    "the GenCAT-style class-attribute correlation matrix H."
                ),
                "density_policy": (
                    "Probabilities are globally scaled by p_scale after blending "
                    "class-specific H with a global feature prior."
                ),
            },
        }
        save_json(summary_path, payload)

        summary_rows.append(summary_row)
        summary_payloads.append(payload)
        cleanup_runtime_memory()

    save_json(
        output_dir / "density_sweep_summary.json",
        {
            "config": {
                "base_config": str(config_path),
                "p_values": [float(value) for value in args.p_values],
                "global_mix": float(args.global_mix),
            },
            "runs": summary_payloads,
        },
    )
    write_summary_csv(output_dir / "density_sweep_summary.csv", summary_rows)

    sorted_rows = sorted(summary_rows, key=lambda row: float(row["p_scale"]))
    p_values = [float(row["p_scale"]) for row in sorted_rows]
    densities = [float(row["observed_feature_density"]) for row in sorted_rows]
    nmis = [float(row["nmi"]) for row in sorted_rows]
    aris = [float(row["ari"]) for row in sorted_rows]
    accuracies = [float(row["clustering_accuracy"]) for row in sorted_rows]

    plot_curve(
        p_values,
        densities,
        output_dir / "feature_density_vs_p.png",
        x_label="p scale",
        y_label="Observed feature density",
        title="Observed attribute density across Bernoulli scaling factors",
    )
    plot_curve(
        densities,
        nmis,
        output_dir / "nmi_vs_density.png",
        x_label="Observed feature density",
        y_label="NMI",
        title="DensGNN NMI across attribute densification levels",
    )
    plot_curve(
        densities,
        aris,
        output_dir / "ari_vs_density.png",
        x_label="Observed feature density",
        y_label="ARI",
        title="DensGNN ARI across attribute densification levels",
    )
    plot_curve(
        densities,
        accuracies,
        output_dir / "accuracy_vs_density.png",
        x_label="Observed feature density",
        y_label="Clustering accuracy",
        title="DensGNN clustering accuracy across attribute densification levels",
    )

    print(f"DensGNN density experiment finished. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
