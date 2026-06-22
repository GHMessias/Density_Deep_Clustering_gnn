from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_benchmark.config.yaml import apply_overrides, load_yaml_config  # noqa: E402
from graph_benchmark.runner import run_from_config  # noqa: E402
from graph_benchmark.utils.seed import set_random_seed  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in this project, but keep runner robust.
    torch = None


DEFAULT_SEEDS = [str(torch.randint(0, 1000000, size=(1,)).item()) for _ in range(5)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run every YAML inside configs/final_inputs across a shared list of seeds.",
    )
    parser.add_argument(
        "--config-dir",
        default=str(PROJECT_ROOT / "configs" / "final_inputs"),
        help="Directory containing the final benchmark YAML files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.yaml",
        help="Glob pattern used to select YAML files inside the config directory.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional root directory for outputs. If omitted, results are saved under "
            "results/<config-dir-name>/..."
        ),
    )
    parser.add_argument(
        "--exclude-substrings",
        nargs="*",
        default=[],
        help="Optional substrings used to skip matching config filenames.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[str(seed) for seed in DEFAULT_SEEDS],
        help="Seeds applied to every selected config. Use 'None' to keep the YAML/default behavior.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose metrics.json already exists for the requested seed.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop as soon as one run fails.",
    )
    return parser


def build_seed_output_dir(
    base_output_dir: str | None,
    seed_label: str,
    config_path: Path,
    *,
    output_root: str | None,
    config_dir_name: str,
) -> str:
    if output_root:
        return str(Path(output_root) / config_path.stem / f"seed_{seed_label}")

    if base_output_dir:
        base_path = Path(base_output_dir)
        return str(Path("results") / config_dir_name / config_path.stem / f"seed_{seed_label}")

    fallback = Path("results") / config_dir_name / config_path.stem / f"seed_{seed_label}"
    return str(fallback)


def parse_seed_value(raw_seed: str | int | None) -> int | None:
    if raw_seed is None:
        return None
    if isinstance(raw_seed, int):
        return raw_seed

    normalized = str(raw_seed).strip().lower()
    if normalized in {"none", "null"}:
        return None
    return int(normalized)


def cleanup_runtime_memory() -> None:
    """Release Python and CUDA memory between long benchmark runs.

    The final benchmark runner executes many experiments sequentially in a
    single Python process. Without an explicit cleanup step, Python references,
    PyTorch's CUDA caching allocator, and inter-process CUDA handles may linger
    between runs and make GPU memory usage look like a leak. This helper keeps
    the runner lightweight without changing the training code of each model.
    """

    gc.collect()

    if torch is None or not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        # Some CUDA runtimes may not expose IPC cleanup in every context.
        pass


def main() -> None:
    args = build_parser().parse_args()
    config_dir = Path(args.config_dir).expanduser()
    config_dir_name = config_dir.name
    config_paths = sorted(config_dir.glob(args.pattern))
    if args.exclude_substrings:
        config_paths = [
            path
            for path in config_paths
            if not any(fragment in path.name for fragment in args.exclude_substrings)
        ]

    if not config_paths:
        raise FileNotFoundError(f"No YAML files found in {config_dir} matching pattern {args.pattern!r}.")

    failures: list[str] = []

    for config_path in config_paths:
        none_counter = 0
        for raw_seed in args.seeds:
            seed = parse_seed_value(raw_seed)
            if seed is None:
                none_counter += 1
                seed = random.SystemRandom().randint(0, 2**31 - 1)
                seed_label = f"none_{none_counter}"
            else:
                seed_label = str(seed)
            base_config = None
            config = None
            results = None
            metrics = None
            try:
                base_config = load_yaml_config(config_path)
                base_output_dir = base_config.get("output", {}).get("dir")
                seed_output_dir = build_seed_output_dir(
                    base_output_dir,
                    seed_label,
                    config_path,
                    output_root=args.output_root,
                    config_dir_name=config_dir_name,
                )
                metrics_path = PROJECT_ROOT / seed_output_dir / "metrics.json"

                if args.skip_existing and metrics_path.exists():
                    print(f"[skip] {config_path.name} seed={seed_label} -> {metrics_path}")
                    continue

                overrides = [
                    f"run.seed={seed}",
                    f"algorithm.params.random_state={seed}",
                    f"output.dir={seed_output_dir}",
                ]
                config = apply_overrides(base_config, overrides)

                set_random_seed(seed)
                results = run_from_config(config)
                metrics = results.get("metrics", {})
                nmi = metrics.get("nmi")
                if isinstance(nmi, (int, float)):
                    print(f"[done] {config_path.name} seed={seed_label} actual_seed={seed} nmi={nmi:.4f}")
                else:
                    print(f"[done] {config_path.name} seed={seed_label} actual_seed={seed}")
            except Exception as exc:  # noqa: BLE001
                message = f"{config_path.name} seed={seed_label}: {exc}"
                failures.append(message)
                print(f"[fail] {message}")
                if args.fail_fast:
                    break
            finally:
                results = None
                config = None
                base_config = None
                cleanup_runtime_memory()

        if failures and args.fail_fast:
            break

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
