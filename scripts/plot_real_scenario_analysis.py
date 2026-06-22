from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "real_scenario_benchmark"


FAMILY_SPECS = [
    ("real_scenario_arga_kmeans_k", "ARGA + KMeans", "#1f77b4"),
    ("real_scenario_arga_hdbscan", "ARGA + HDBSCAN", "#4c78a8"),
    ("real_scenario_gae_kmeans_k", "GAE + KMeans", "#ff7f0e"),
    ("real_scenario_gae_hdbscan", "GAE + HDBSCAN", "#f58518"),
    ("real_scenario_densgnn_core", "DensGNN Core", "#2ca02c"),
    ("real_scenario_densgnn_border", "DensGNN Border", "#d62728"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate visual analysis plots for the real_scenario_benchmark results."
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing the saved runs. Default: results/real_scenario_benchmark",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the analysis figures will be saved. Default: <results-dir>/analysis",
    )
    return parser


def family_from_group(group_name: str) -> tuple[str, str]:
    for prefix, label, color in FAMILY_SPECS:
        if group_name.startswith(prefix):
            return label, color
    return group_name, "#7f7f7f"


def extract_k(group_name: str) -> int | None:
    marker = "_k"
    if marker not in group_name:
        return None
    suffix = group_name.rsplit(marker, maxsplit=1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def infer_group_name(metrics_path: Path) -> str:
    parent_name = metrics_path.parent.name
    if parent_name.startswith("seed_") and len(metrics_path.parents) >= 2:
        return metrics_path.parents[1].name
    return parent_name


def load_runs(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_files = sorted(results_dir.glob("**/metrics.json"))
    for metrics_path in metric_files:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        dataset = payload.get("dataset", {})
        group_name = infer_group_name(metrics_path)
        family_name, family_color = family_from_group(group_name)

        rows.append(
            {
                "metrics_path": str(metrics_path),
                "group_name": group_name,
                "seed_name": metrics_path.parent.name,
                "family": family_name,
                "color": family_color,
                "k": extract_k(group_name),
                "dataset_name": dataset.get("name"),
                "target_num_classes": dataset.get("num_classes"),
                "nmi": metrics.get("nmi"),
                "ari": metrics.get("ari"),
                "silhouette": metrics.get("silhouette"),
                "modularity": metrics.get("modularity"),
                "num_clusters_found": metrics.get("num_clusters_found"),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"No metrics.json files found under {results_dir}")

    numeric_columns = ["k", "target_num_classes", "nmi", "ari", "silhouette", "modularity", "num_clusters_found"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def save_raw_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        "family",
        "group_name",
        "seed_name",
        "k",
        "nmi",
        "ari",
        "silhouette",
        "modularity",
        "num_clusters_found",
    ]
    frame[columns].to_csv(output_dir / "run_level_metrics.csv", index=False)


def save_nmi_runs(frame: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        "family",
        "group_name",
        "seed_name",
        "k",
        "nmi",
    ]
    nmi_frame = frame[columns].copy().sort_values(["family", "group_name", "seed_name"])
    nmi_frame.to_csv(output_dir / "nmi_runs.csv", index=False)


def save_family_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        frame.groupby("family", dropna=False)[["nmi", "ari", "silhouette", "modularity", "num_clusters_found"]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    summary.to_csv(output_dir / "family_summary.csv")


def add_jitter(values: list[float], amount: float = 0.06) -> list[float]:
    if not values:
        return values
    # Deterministic jitter so the same figure is reproducible without random seeds.
    if len(values) == 1:
        return [values[0]]
    return [
        value + amount * (((index / max(1, len(values) - 1)) * 2.0) - 1.0)
        for index, value in enumerate(values)
    ]


def plot_metric_distributions(frame: pd.DataFrame, output_dir: Path) -> None:
    families = [label for _, label, _ in FAMILY_SPECS if label in set(frame["family"])]
    color_by_family = {label: color for _, label, color in FAMILY_SPECS}
    metrics = [
        ("nmi", "NMI"),
        ("silhouette", "Silhouette"),
        ("modularity", "Modularity"),
        ("num_clusters_found", "Clusters Found"),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for axis, (metric_key, metric_label) in zip(axes, metrics, strict=True):
        series_by_family = [
            frame.loc[frame["family"] == family, metric_key].dropna().tolist()
            for family in families
        ]
        positions = list(range(1, len(families) + 1))
        axis.boxplot(series_by_family, positions=positions, widths=0.55, patch_artist=True)

        for patch, family in zip(axis.artists, families, strict=False):
            patch.set_facecolor(color_by_family.get(family, "#cccccc"))
            patch.set_alpha(0.25)

        for position, family, values in zip(positions, families, series_by_family, strict=True):
            x_values = add_jitter([float(position)] * len(values))
            axis.scatter(
                x_values,
                values,
                s=28,
                alpha=0.7,
                color=color_by_family.get(family, "#666666"),
                edgecolors="white",
                linewidths=0.5,
            )

        axis.set_title(metric_label)
        axis.set_xticks(positions)
        axis.set_xticklabels(families, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Run-Level Metric Distributions", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_dir / "metric_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_nmi_boxplot(frame: pd.DataFrame, output_dir: Path) -> None:
    families = [label for _, label, _ in FAMILY_SPECS if label in set(frame["family"])]
    color_by_family = {label: color for _, label, color in FAMILY_SPECS}
    series_by_family = [
        frame.loc[frame["family"] == family, "nmi"].dropna().tolist()
        for family in families
    ]

    figure, axis = plt.subplots(figsize=(11, 6.5))
    boxplot = axis.boxplot(series_by_family, positions=list(range(1, len(families) + 1)), widths=0.55, patch_artist=True)

    for patch, family in zip(boxplot["boxes"], families, strict=True):
        patch.set_facecolor(color_by_family.get(family, "#cccccc"))
        patch.set_alpha(0.35)

    for position, family, values in zip(range(1, len(families) + 1), families, series_by_family, strict=True):
        x_values = add_jitter([float(position)] * len(values))
        axis.scatter(
            x_values,
            values,
            s=34,
            alpha=0.75,
            color=color_by_family.get(family, "#666666"),
            edgecolors="white",
            linewidths=0.5,
        )

    axis.set_title("NMI Distribution by Model")
    axis.set_ylabel("NMI")
    axis.set_xticks(list(range(1, len(families) + 1)))
    axis.set_xticklabels(families, rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "nmi_boxplot.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_tradeoff_bubble(frame: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        frame.groupby("family", dropna=False)
        .agg(
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            modularity_mean=("modularity", "mean"),
            modularity_std=("modularity", "std"),
            clusters_mean=("num_clusters_found", "mean"),
        )
        .reset_index()
    )

    figure, axis = plt.subplots(figsize=(10, 7))
    for _, row in summary.iterrows():
        family = row["family"]
        color = next((color for _, label, color in FAMILY_SPECS if label == family), "#666666")
        nmi_mean = row["nmi_mean"]
        mod_mean = row["modularity_mean"]
        nmi_std = row["nmi_std"]
        mod_std = row["modularity_std"]
        cluster_mean = row["clusters_mean"]

        size = 50 + max(0.0, float(cluster_mean)) * 18.0
        axis.errorbar(
            nmi_mean,
            mod_mean,
            xerr=0.0 if pd.isna(nmi_std) else nmi_std,
            yerr=0.0 if pd.isna(mod_std) else mod_std,
            fmt="none",
            ecolor=color,
            alpha=0.4,
            capsize=3,
        )
        axis.scatter(nmi_mean, mod_mean, s=size, color=color, alpha=0.75, edgecolors="black", linewidths=0.7)
        axis.annotate(
            family,
            (nmi_mean, mod_mean),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    axis.set_xlabel("Mean NMI")
    axis.set_ylabel("Mean Modularity")
    axis.set_title("Family Trade-off: NMI vs Modularity\nBubble size = mean number of clusters")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "tradeoff_bubble.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_kmeans_k_sweep(frame: pd.DataFrame, output_dir: Path) -> None:
    subset = frame[frame["family"].isin(["ARGA + KMeans", "GAE + KMeans"])].copy()
    if subset.empty:
        return

    grouped = (
        subset.groupby(["family", "k"], dropna=False)
        .agg(
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            silhouette_mean=("silhouette", "mean"),
            silhouette_std=("silhouette", "std"),
            modularity_mean=("modularity", "mean"),
            modularity_std=("modularity", "std"),
        )
        .reset_index()
        .sort_values(["family", "k"])
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    metric_specs = [
        ("nmi_mean", "nmi_std", "NMI"),
        ("silhouette_mean", "silhouette_std", "Silhouette"),
        ("modularity_mean", "modularity_std", "Modularity"),
    ]

    for axis, (mean_key, std_key, metric_label) in zip(axes, metric_specs, strict=True):
        for family in ["ARGA + KMeans", "GAE + KMeans"]:
            family_rows = grouped[grouped["family"] == family]
            if family_rows.empty:
                continue
            color = next((color for _, label, color in FAMILY_SPECS if label == family), "#666666")
            axis.errorbar(
                family_rows["k"],
                family_rows[mean_key],
                yerr=family_rows[std_key].fillna(0.0),
                marker="o",
                linewidth=2,
                capsize=3,
                color=color,
                label=family,
            )

        axis.set_title(f"{metric_label} by $k$")
        axis.set_xlabel("$k$")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Metric value")
    axes[-1].legend(loc="best")
    figure.suptitle("KMeans Families Across Tested Values of $k$", fontsize=13)
    figure.tight_layout()
    figure.savefig(output_dir / "kmeans_k_sweep.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_clusters_vs_nmi(frame: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))
    target_k = frame["target_num_classes"].dropna().iloc[0] if frame["target_num_classes"].notna().any() else None

    for _, family, color in FAMILY_SPECS:
        family_rows = frame[frame["family"] == family]
        if family_rows.empty:
            continue
        axis.scatter(
            family_rows["num_clusters_found"],
            family_rows["nmi"],
            s=55,
            color=color,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
            label=family,
        )

    if target_k is not None and not math.isnan(float(target_k)):
        axis.axvline(float(target_k), color="black", linestyle="--", linewidth=1.2, alpha=0.7, label=f"True k={int(target_k)}")

    axis.set_xlabel("Clusters found")
    axis.set_ylabel("NMI")
    axis.set_title("Run-Level NMI vs Number of Clusters Found")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "clusters_vs_nmi.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()

    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else results_dir / "analysis"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_runs(results_dir)
    save_raw_summary(frame, output_dir)
    save_nmi_runs(frame, output_dir)
    save_family_summary(frame, output_dir)
    plot_nmi_boxplot(frame, output_dir)
    plot_metric_distributions(frame, output_dir)
    plot_tradeoff_bubble(frame, output_dir)
    plot_kmeans_k_sweep(frame, output_dir)
    plot_clusters_vs_nmi(frame, output_dir)

    print(f"Results dir: {results_dir}")
    print(f"Runs loaded: {len(frame)}")
    print(f"Analysis saved to: {output_dir}")
    print("Generated files:")
    for filename in [
        "run_level_metrics.csv",
        "nmi_runs.csv",
        "family_summary.csv",
        "nmi_boxplot.png",
        "metric_distributions.png",
        "tradeoff_bubble.png",
        "kmeans_k_sweep.png",
        "clusters_vs_nmi.png",
    ]:
        print(f"- {output_dir / filename}")


if __name__ == "__main__":
    main()
