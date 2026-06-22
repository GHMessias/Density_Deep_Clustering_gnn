from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - depends on the local environment.
    raise SystemExit(
        "The analysis script now depends on pandas. Install it in the project venv with "
        "'./venv/bin/pip install pandas' and run the command again."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
MODEL_SPECS = [
    ("real_scenario_arga_kmeans", "ARGA + KMeans (best $k$)", "$X,A$", 50),
    ("real_scenario_arga_hdbscan", "ARGA + HDBSCAN", "$X,A$", 51),
    ("real_scenario_gae_kmeans", "GAE + KMeans (best $k$)", "$X,A$", 60),
    ("real_scenario_gae_hdbscan", "GAE + HDBSCAN", "$X,A$", 61),
    ("real_scenario_densgnn_border", "DensGNN (Border)", "$X,A$", 80),
    ("real_scenario_densgnn_core", "DensGNN (Core)", "$X,A$", 81),
    ("ollivier_ricci_community", "ORC Community \\cite{Ni2019community}", "$A$", 10),
    ("node2vec_kmeans", "Node2Vec", "$A$", 20),
    ("kmeans_features", "KMeans", "$X$", 30),
    ("dgcss_bc", "DGCSS (BC) \\cite{Filho2026deep}", "$X,A$", 40),
    ("dgcss_kmeans", "DGCSS (KMeans) \\cite{Filho2026deep}", "$X,A$", 45),
    ("arga_kmeans", "ARGA \\cite{pan2019adversarially}", "$X,A$", 50),
    ("gae_kmeans", "GAE \\cite{kipf2016variational}", "$X,A$", 60),
    ("densgnn2_border", "DensGNN2 (Border)", "$X,A$", 70),
    ("densgnn2_core", "DensGNN2 (Core)", "$X,A$", 71),
    ("densgnn2", "DNENC \\cite{Wang2022deep}", "$X,A$", 72),
    ("densgnn_border", "DensGNN (Border)", "$X,A$", 80),
    ("densgnn_core", "DensGNN (Core)", "$X,A$", 81),
    ("densgnn", "DensGNN", "$X,A$", 82),
]
PREFERRED_METRIC_ORDER = [
    "nmi",
    "num_clusters_found",
    "silhouette",
    "modularity",
    "ari",
    "clustering_accuracy",
    "purity",
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
    "node2vec_loss",
]
SUMMARY_METRICS = [
    "nmi",
    "num_clusters_found",
    "silhouette",
    "modularity",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results either by dataset ranking or by aggregated seed runs.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset directory under results/. Example: cora. Used by the legacy ranking mode.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing experiment subfolders with seed_x/metrics.json files.",
    )
    parser.add_argument(
        "--sort-by",
        default="nmi",
        help="Metric used to rank runs or aggregated groups. Default: nmi",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many runs to show in the legacy global ranking mode. Default: 10",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Emit the aggregated table in LaTeX format.",
    )
    parser.add_argument(
        "--collapse-real-scenario",
        action="store_true",
        help=(
            "Collapse real_scenario_benchmark families into 6 algorithms. "
            "See --collapse-real-scenario-mode to choose whether KMeans families use the "
            "best k or the average across all runs."
        ),
    )
    parser.add_argument(
        "--collapse-real-scenario-mode",
        choices=("best-k", "all-runs"),
        default="best-k",
        help=(
            "How to collapse KMeans families in real_scenario_benchmark. "
            "'best-k' keeps the k with the best mean --sort-by. "
            "'all-runs' averages all runs across every tested k."
        ),
    )
    return parser


def load_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    algorithm = payload.get("algorithm", {})
    return {
        "run_name": path.parent.name,
        "group_name": infer_group_name(path),
        "metrics_path": path,
        "output_dir": payload.get("artifacts", {}).get("output_dir", str(path.parent)),
        "algorithm_name": algorithm.get("name", "<unknown>"),
        "metrics": metrics,
    }


def infer_group_name(path: Path) -> str:
    parent_name = path.parent.name
    if parent_name.startswith("seed_") and len(path.parents) >= 2:
        return path.parents[1].name
    return parent_name


def fmt_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "-"
    return str(value)


def fmt_mean_std(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None:
        return "-"
    if std_value is None:
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} +- {truncate_decimals(std_value, 2):.2f}"


def fmt_mean_std_latex(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None:
        return "-"
    if std_value is None:
        return f"{mean_value:.4f}"
    return f"${mean_value:.4f} \\pm {truncate_decimals(std_value, 2):.2f}$"


def fmt_cluster_mean_std(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None:
        return "-"
    cluster_mean = int(round(mean_value))
    if std_value is None:
        return str(cluster_mean)
    return f"{cluster_mean} +- {truncate_decimals(std_value, 2):.2f}"


def fmt_cluster_mean_std_latex(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None:
        return "-"
    cluster_mean = int(round(mean_value))
    if std_value is None:
        return str(cluster_mean)
    return f"${cluster_mean} \\pm {truncate_decimals(std_value, 2):.2f}$"


def truncate_decimals(value: float, decimals: int) -> float:
    factor = 10**decimals
    return math.trunc(value * factor) / factor


def metric_value(run: dict[str, Any], metric_name: str) -> float:
    value = run["metrics"].get(metric_name)
    if isinstance(value, (int, float)):
        return float(value)
    return float("-inf")


def numeric_metric_names(runs: list[dict[str, Any]]) -> list[str]:
    present = {
        key
        for run in runs
        for key, value in run["metrics"].items()
        if isinstance(value, (int, float))
    }
    preferred = [name for name in PREFERRED_METRIC_ORDER if name in present]
    remaining = sorted(present.difference(preferred))
    return preferred + remaining


def aggregate_groups(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["group_name"], []).append(run)

    aggregated_rows: list[dict[str, Any]] = []
    metric_names = numeric_metric_names(runs)

    for group_name, group_runs in sorted(grouped.items()):
        first_run = group_runs[0]
        aggregated_metrics: dict[str, dict[str, float | int]] = {}

        for metric_name in metric_names:
            values = [
                float(run["metrics"][metric_name])
                for run in group_runs
                if isinstance(run["metrics"].get(metric_name), (int, float))
            ]
            if not values:
                continue

            aggregated_metrics[metric_name] = {
                "mean": float(mean(values)),
                "std": float(stdev(values)) if len(values) > 1 else 0.0,
                "count": len(values),
            }

        aggregated_rows.append(
            {
                "group_name": group_name,
                "algorithm_name": first_run["algorithm_name"],
                "num_runs": len(group_runs),
                "metrics": aggregated_metrics,
            }
        )

    return aggregated_rows


def aggregated_metric_value(row: dict[str, Any], metric_name: str) -> float:
    metric_summary = row["metrics"].get(metric_name)
    if isinstance(metric_summary, dict):
        mean_value = metric_summary.get("mean")
        if isinstance(mean_value, (int, float)):
            return float(mean_value)
    return float("-inf")


def real_scenario_family_name(group_name: str) -> str | None:
    if group_name.startswith("real_scenario_arga_kmeans_k"):
        return "real_scenario_arga_kmeans"
    if group_name == "real_scenario_arga_hdbscan":
        return "real_scenario_arga_hdbscan"
    if group_name.startswith("real_scenario_gae_kmeans_k"):
        return "real_scenario_gae_kmeans"
    if group_name == "real_scenario_gae_hdbscan":
        return "real_scenario_gae_hdbscan"
    if group_name == "real_scenario_densgnn_core":
        return "real_scenario_densgnn_core"
    if group_name == "real_scenario_densgnn_border":
        return "real_scenario_densgnn_border"
    return None


def extract_k_from_group_name(group_name: str) -> int | None:
    marker = "_k"
    if marker not in group_name:
        return None
    suffix = group_name.rsplit(marker, maxsplit=1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def collapse_real_scenario_rows(
    aggregated_rows: list[dict[str, Any]],
    sort_by: str,
) -> list[dict[str, Any]]:
    family_groups: dict[str, list[dict[str, Any]]] = {}
    untouched_rows: list[dict[str, Any]] = []

    for row in aggregated_rows:
        family_name = real_scenario_family_name(row["group_name"])
        if family_name is None:
            untouched_rows.append(row)
            continue
        family_groups.setdefault(family_name, []).append(row)

    collapsed_rows: list[dict[str, Any]] = []
    for family_name, rows in family_groups.items():
        if family_name in {"real_scenario_arga_kmeans", "real_scenario_gae_kmeans"}:
            best_row = max(rows, key=lambda row: aggregated_metric_value(row, sort_by))
            selected_k = extract_k_from_group_name(best_row["group_name"])
            collapsed_metrics = dict(best_row["metrics"])
            if selected_k is not None:
                collapsed_metrics["selected_k"] = {
                    "mean": float(selected_k),
                    "std": 0.0,
                    "count": 1,
                }
            collapsed_rows.append(
                {
                    "group_name": family_name,
                    "algorithm_name": best_row["algorithm_name"],
                    "num_runs": best_row["num_runs"],
                    "metrics": collapsed_metrics,
                }
            )
            continue

        representative_row = max(rows, key=lambda row: aggregated_metric_value(row, sort_by))
        collapsed_rows.append(
            {
                "group_name": family_name,
                "algorithm_name": representative_row["algorithm_name"],
                "num_runs": representative_row["num_runs"],
                "metrics": representative_row["metrics"],
            }
        )

    return untouched_rows + collapsed_rows


def collapse_real_scenario_runs_all_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed_runs: list[dict[str, Any]] = []
    for run in runs:
        family_name = real_scenario_family_name(run["group_name"])
        if family_name is None:
            collapsed_runs.append(run)
            continue
        collapsed_run = dict(run)
        collapsed_run["group_name"] = family_name
        collapsed_runs.append(collapsed_run)
    return collapsed_runs


def print_legacy_table(runs: list[dict[str, Any]], title: str, sort_by: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not runs:
        print("Nenhum resultado encontrado.")
        return

    header = (
        f"{'#':>3}  "
        f"{'run':<28}  "
        f"{'algorithm':<24}  "
        f"{'NMI':>8}  "
        f"{'Clusters':>10}  "
        f"{'Silh':>8}  "
        f"{'Mod':>8}"
    )
    print(header)
    print("-" * len(header))

    for index, run in enumerate(runs, start=1):
        metrics = run["metrics"]
        print(
            f"{index:>3}  "
            f"{run['run_name']:<28}  "
            f"{run['algorithm_name']:<24}  "
            f"{fmt_metric(metrics.get('nmi')):>8}  "
            f"{fmt_metric(metrics.get('num_clusters_found')):>10}  "
            f"{fmt_metric(metrics.get('silhouette')):>8}  "
            f"{fmt_metric(metrics.get('modularity')):>8}"
        )


def prettify_metric_name(metric_name: str) -> str:
    return metric_name.upper()


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def infer_model_spec(group_name: str) -> tuple[str, str, int]:
    for suffix, display_name, source, order in MODEL_SPECS:
        if group_name.endswith(suffix):
            return display_name, source, order
    return latex_escape(group_name), "$X,A$", 999


def infer_dataset_label(results_dir: Path) -> tuple[str, str]:
    dataset_key = results_dir.name.replace("_benchmarks", "").replace("_results", "").lower()
    dataset_name_map = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
        "polblogs": "PolBlogs",
        "twitch": "Twitch",
    }
    dataset_label = dataset_name_map.get(dataset_key, dataset_key.title())
    table_label = f"tab:{dataset_key}_clustering_results"
    return dataset_label, table_label


def build_aggregated_dataframe(rows: list[dict[str, Any]], metric_names: list[str]) -> pd.DataFrame:
    table_rows: list[dict[str, Any]] = []
    for row in rows:
        output_row: dict[str, Any] = {
            "Experiment": row["group_name"],
        }
        selected_k_summary = row["metrics"].get("selected_k")
        if isinstance(selected_k_summary, dict):
            output_row["SELECTED_K"] = fmt_cluster_mean_std(
                selected_k_summary.get("mean"),
                selected_k_summary.get("std"),
            )
        for metric_name in metric_names:
            metric_summary = row["metrics"].get(metric_name)
            mean_value = metric_summary.get("mean") if isinstance(metric_summary, dict) else None
            std_value = metric_summary.get("std") if isinstance(metric_summary, dict) else None
            if metric_name == "num_clusters_found":
                output_row[prettify_metric_name(metric_name)] = fmt_cluster_mean_std(mean_value, std_value)
            else:
                output_row[prettify_metric_name(metric_name)] = fmt_mean_std(mean_value, std_value)
        table_rows.append(output_row)

    return pd.DataFrame(table_rows)


def print_aggregated_table(rows: list[dict[str, Any]], metric_names: list[str]) -> None:
    if not rows:
        print("Nenhum resultado agregado encontrado.")
        return

    dataframe = build_aggregated_dataframe(rows, metric_names)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(dataframe.to_string(index=False))


def build_latex_table(rows: list[dict[str, Any]], metric_names: list[str]) -> str:
    raise NotImplementedError("Use build_paper_latex_table for the paper-formatted LaTeX output.")


def format_metric_for_paper(
    metric_name: str,
    mean_value: float | None,
    std_value: float | None,
    *,
    highlight: str | None = None,
) -> str:
    if metric_name == "num_clusters_found":
        text = fmt_cluster_mean_std_latex(mean_value, std_value)
    else:
        text = fmt_mean_std_latex(mean_value, std_value)

    if text.startswith("$") and text.endswith("$"):
        core_text = text[1:-1]
    else:
        core_text = text

    if highlight == "best":
        return f"$\\mathbf{{{core_text}}}$" if text != "-" else text
    if highlight == "second":
        return f"$\\underline{{{core_text}}}$" if text != "-" else text
    return text


def build_paper_latex_table(rows: list[dict[str, Any]], results_dir: Path) -> str:
    if not rows:
        return ""

    ordered_rows: list[dict[str, Any]] = []
    for row in rows:
        model_name, source, order = infer_model_spec(row["group_name"])
        ordered_rows.append(
            {
                "row": row,
                "model_name": model_name,
                "source": source,
                "order": order,
            }
        )

    ordered_rows.sort(key=lambda item: (item["order"], item["model_name"]))

    nmi_rank = sorted(
        [
            (
                item["row"]["group_name"],
                item["row"]["metrics"].get("nmi", {}).get("mean"),
            )
            for item in ordered_rows
            if isinstance(item["row"]["metrics"].get("nmi"), dict)
            and isinstance(item["row"]["metrics"]["nmi"].get("mean"), (int, float))
        ],
        key=lambda pair: float(pair[1]),
        reverse=True,
    )
    best_group = nmi_rank[0][0] if len(nmi_rank) >= 1 else None
    second_group = nmi_rank[1][0] if len(nmi_rank) >= 2 else None

    dataset_label, table_label = infer_dataset_label(results_dir)

    body_lines: list[str] = []
    densgnn_separator_inserted = False
    for item in ordered_rows:
        row = item["row"]
        if item["model_name"].startswith("DensGNN") and not densgnn_separator_inserted:
            body_lines.append("\\hline")
            densgnn_separator_inserted = True

        metrics = row["metrics"]
        selected_k_summary = metrics.get("selected_k")
        nmi_mean = metrics.get("nmi", {}).get("mean") if isinstance(metrics.get("nmi"), dict) else None
        nmi_std = metrics.get("nmi", {}).get("std") if isinstance(metrics.get("nmi"), dict) else None
        cluster_mean = (
            metrics.get("num_clusters_found", {}).get("mean")
            if isinstance(metrics.get("num_clusters_found"), dict)
            else None
        )
        cluster_std = (
            metrics.get("num_clusters_found", {}).get("std")
            if isinstance(metrics.get("num_clusters_found"), dict)
            else None
        )
        silhouette_mean = (
            metrics.get("silhouette", {}).get("mean")
            if isinstance(metrics.get("silhouette"), dict)
            else None
        )
        silhouette_std = (
            metrics.get("silhouette", {}).get("std")
            if isinstance(metrics.get("silhouette"), dict)
            else None
        )
        modularity_mean = (
            metrics.get("modularity", {}).get("mean")
            if isinstance(metrics.get("modularity"), dict)
            else None
        )
        modularity_std = (
            metrics.get("modularity", {}).get("std")
            if isinstance(metrics.get("modularity"), dict)
            else None
        )

        highlight = None
        if row["group_name"] == best_group:
            highlight = "best"
        elif row["group_name"] == second_group:
            highlight = "second"

        model_name = item["model_name"]
        if isinstance(selected_k_summary, dict):
            selected_k_mean = selected_k_summary.get("mean")
            if isinstance(selected_k_mean, (int, float)):
                model_name = f"{model_name} [best $k$={int(round(float(selected_k_mean)))}]"

        nmi_text = format_metric_for_paper("nmi", nmi_mean, nmi_std, highlight=highlight)
        cluster_text = format_metric_for_paper("num_clusters_found", cluster_mean, cluster_std)
        silhouette_text = format_metric_for_paper("silhouette", silhouette_mean, silhouette_std)
        modularity_text = format_metric_for_paper("modularity", modularity_mean, modularity_std)

        body_lines.append(
            f"{model_name} & {item['source']} & {nmi_text} & {cluster_text} & "
            f"{silhouette_text} & {modularity_text} \\\\"
        )

    table_lines = [
        "\\begin{table}",
        f"\\caption{{{dataset_label} dataset clustering performance}}",
        f"\\label{{{table_label}}}",
        "\\footnotesize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|c|c|c|c|c|}",
        "\\hline",
        "\\textbf{Model} & \\textbf{Source} & \\textbf{NMI} & $k$ & \\textbf{Sil.} & \\textbf{Mod.} \\\\",
        "\\hline",
        *body_lines,
        "\\hline",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ]
    return "\n".join(table_lines)


def run_legacy_dataset_mode(args: argparse.Namespace) -> None:
    dataset_name = args.dataset or "cora"
    dataset_dir = RESULTS_ROOT / dataset_name

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    metric_files = sorted(dataset_dir.glob("*/metrics.json"))
    runs = [load_run(path) for path in metric_files]

    if not runs:
        raise SystemExit(f"No metrics.json files found under {dataset_dir}")

    available_metrics = numeric_metric_names(runs)
    if args.sort_by not in available_metrics:
        available = ", ".join(available_metrics)
        raise SystemExit(
            f"Metric '{args.sort_by}' not found for dataset '{dataset_name}'. "
            f"Available numeric metrics: {available}"
        )

    ranked_runs = sorted(runs, key=lambda run: metric_value(run, args.sort_by), reverse=True)
    best_per_algorithm: dict[str, dict[str, Any]] = {}
    for run in ranked_runs:
        algorithm_name = run["algorithm_name"]
        if algorithm_name not in best_per_algorithm:
            best_per_algorithm[algorithm_name] = run

    print(f"Dataset: {dataset_name}")
    print(f"Runs found: {len(runs)}")
    print(f"Ranking metric: {args.sort_by}")

    print_legacy_table(ranked_runs[: max(1, args.top)], "Global Ranking", args.sort_by)

    grouped_runs = sorted(
        best_per_algorithm.values(),
        key=lambda run: metric_value(run, args.sort_by),
        reverse=True,
    )
    print_legacy_table(grouped_runs, "Best Run Per Algorithm", args.sort_by)

    best_run = ranked_runs[0]
    print("\nBest overall run")
    print("----------------")
    print(f"Run: {best_run['run_name']}")
    print(f"Algorithm: {best_run['algorithm_name']}")
    print(f"Metrics path: {best_run['metrics_path']}")
    print(f"Output dir: {best_run['output_dir']}")
    for metric_name in ("nmi", "num_clusters_found", "silhouette", "modularity"):
        if metric_name in best_run["metrics"]:
            print(f"{metric_name}: {fmt_metric(best_run['metrics'][metric_name])}")


def run_aggregated_results_mode(args: argparse.Namespace) -> None:
    if args.results_dir is None:
        raise SystemExit("The aggregated mode requires --results-dir.")

    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    metric_files = sorted(results_dir.glob("**/metrics.json"))
    runs = [load_run(path) for path in metric_files]

    if not runs:
        raise SystemExit(f"No metrics.json files found under {results_dir}")

    aggregated_rows = aggregate_groups(runs)
    available_metric_names = numeric_metric_names(runs)
    metric_names = [metric for metric in SUMMARY_METRICS if metric in available_metric_names]

    if args.sort_by not in available_metric_names:
        available = ", ".join(available_metric_names)
        raise SystemExit(
            f"Metric '{args.sort_by}' not found under '{results_dir}'. "
            f"Available numeric metrics: {available}"
        )

    if args.collapse_real_scenario:
        if args.collapse_real_scenario_mode == "all-runs":
            collapsed_runs = collapse_real_scenario_runs_all_runs(runs)
            aggregated_rows = aggregate_groups(collapsed_runs)
        else:
            aggregated_rows = collapse_real_scenario_rows(aggregated_rows, args.sort_by)

    aggregated_rows = sorted(
        aggregated_rows,
        key=lambda row: aggregated_metric_value(row, args.sort_by),
        reverse=True,
    )

    print(f"Results dir: {results_dir}")
    print(f"Seed runs found: {len(runs)}")
    print(f"Experiment groups: {len(aggregated_rows)}")
    print(f"Ranking metric: {args.sort_by}")
    print()
    print_aggregated_table(aggregated_rows, metric_names)

    if args.latex:
        print("\nLaTeX Table")
        print("-----------")
        print(build_paper_latex_table(aggregated_rows, results_dir))


def main() -> None:
    args = build_parser().parse_args()

    if args.results_dir is not None:
        run_aggregated_results_mode(args)
        return

    run_legacy_dataset_mode(args)


if __name__ == "__main__":
    main()
