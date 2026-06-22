from __future__ import annotations

import argparse
import csv
import itertools
import json
from copy import deepcopy
import os
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_benchmark.config.yaml import load_yaml_config, set_dotted_value  # noqa: E402
from graph_benchmark.runner import run_from_config  # noqa: E402
from graph_benchmark.utils.seed import set_random_seed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a sequential grid search for DensGNN using YAML-defined search spaces.",
    )
    parser.add_argument(
        "--search-config",
        default="configs/cora_densgnn_grid_search.yaml",
        help="YAML file describing the base config and the search space.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of combinations to execute.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose metrics.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the combinations that would be executed.",
    )
    return parser


def sanitize_value(value: Any) -> str:
    text = str(value)
    sanitized = []
    for character in text:
        if character.isalnum() or character in {"-", "_"}:
            sanitized.append(character)
        else:
            sanitized.append("-")
    result = "".join(sanitized).strip("-")
    return result or "value"


def compact_key(dotted_key: str) -> str:
    return dotted_key.split(".")[-1]


def generate_grid_runs(search_space: dict[str, list[Any]]) -> list[tuple[dict[str, Any], str]]:
    if not search_space:
        return [({}, "default")]

    keys = list(search_space)
    values_product = itertools.product(*(search_space[key] for key in keys))
    runs: list[tuple[dict[str, Any], str]] = []

    for combination in values_product:
        params = dict(zip(keys, combination))
        parts = [f"{compact_key(key)}-{sanitize_value(value)}" for key, value in params.items()]
        run_name = "__".join(parts)
        runs.append((params, run_name))

    return runs


def metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name)
    if isinstance(value, (int, float)):
        return float(value)
    return float("-inf")


def choose_primary_metric(metrics: dict[str, Any], preferred_metric: str) -> float:
    preferred_value = metric_value(metrics, preferred_metric)
    if preferred_value != float("-inf"):
        return preferred_value
    return metric_value(metrics, "nmi")


def main() -> None:
    args = build_parser().parse_args()
    search_config_path = Path(args.search_config)
    search_config = load_yaml_config(search_config_path)

    base_config_path = PROJECT_ROOT / search_config.get("base_config", "configs/cora_densgnn.yaml")
    base_config = load_yaml_config(base_config_path)
    search_space = search_config.get("search_space", {})
    ranking_metric = str(search_config.get("ranking_metric", "best_eval_nmi"))
    output_root = PROJECT_ROOT / search_config.get("output_root", "results/grid_search/densgnn")
    output_root.mkdir(parents=True, exist_ok=True)

    grid_runs = generate_grid_runs(search_space)
    if args.max_runs is not None:
        grid_runs = grid_runs[: args.max_runs]

    print(f"Base config: {base_config_path}")
    print(f"Search config: {search_config_path}")
    print(f"Ranking metric: {ranking_metric}")
    print(f"Output root: {output_root}")
    print(f"Combinations: {len(grid_runs)}")

    results_summary: list[dict[str, Any]] = []

    for index, (params, run_name) in enumerate(grid_runs, start=1):
        run_output_dir = output_root / run_name
        metrics_path = run_output_dir / "metrics.json"

        print(f"\n[{index}/{len(grid_runs)}] {run_name}")
        for key, value in params.items():
            print(f"  {key} = {value}")

        if args.skip_existing and metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            score = choose_primary_metric(metrics, ranking_metric)
            print(f"  skipped (existing result), score={score:.4f}")
            results_summary.append(
                {
                    "run_name": run_name,
                    "output_dir": str(run_output_dir),
                    "score": score,
                    "metrics": metrics,
                    "params": params,
                }
            )
            continue

        if args.dry_run:
            continue

        config = deepcopy(base_config)
        for key, value in params.items():
            set_dotted_value(config, key, value)
        set_dotted_value(config, "output.dir", str(run_output_dir))
        run_seed = int(config.get("run", {}).get("seed", 42))
        set_random_seed(run_seed)

        results = run_from_config(config)
        metrics = results.get("metrics", {})
        score = choose_primary_metric(metrics, ranking_metric)
        print(
            f"  done: score={score:.4f} "
            f"nmi={metric_value(metrics, 'nmi'):.4f} "
            f"ari={metric_value(metrics, 'ari'):.4f}"
        )

        results_summary.append(
            {
                "run_name": run_name,
                "output_dir": str(run_output_dir),
                "score": score,
                "metrics": metrics,
                "params": params,
            }
        )

    if args.dry_run:
        return

    results_summary.sort(key=lambda item: item["score"], reverse=True)

    summary_json_path = output_root / "grid_search_summary.json"
    summary_json_path.write_text(json.dumps(results_summary, indent=2, sort_keys=False), encoding="utf-8")

    summary_csv_path = output_root / "grid_search_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "rank",
            "run_name",
            "score",
            "best_eval_nmi",
            "nmi",
            "ari",
            "clustering_accuracy",
            "purity",
            "modularity",
            "output_dir",
            "params",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, item in enumerate(results_summary, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "run_name": item["run_name"],
                    "score": item["score"],
                    "best_eval_nmi": item["metrics"].get("best_eval_nmi"),
                    "nmi": item["metrics"].get("nmi"),
                    "ari": item["metrics"].get("ari"),
                    "clustering_accuracy": item["metrics"].get("clustering_accuracy"),
                    "purity": item["metrics"].get("purity"),
                    "modularity": item["metrics"].get("modularity"),
                    "output_dir": item["output_dir"],
                    "params": json.dumps(item["params"], ensure_ascii=True, sort_keys=True),
                }
            )

    print("\nTop runs")
    print("--------")
    for rank, item in enumerate(results_summary[:10], start=1):
        metrics = item["metrics"]
        print(
            f"{rank:>2}. {item['run_name']} | "
            f"score={item['score']:.4f} | "
            f"best_eval_nmi={metric_value(metrics, 'best_eval_nmi'):.4f} | "
            f"nmi={metric_value(metrics, 'nmi'):.4f} | "
            f"ari={metric_value(metrics, 'ari'):.4f}"
        )

    print(f"\nSummary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
