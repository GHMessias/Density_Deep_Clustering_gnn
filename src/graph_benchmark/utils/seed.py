from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_random_seed(seed: int | None) -> None:
    """Set the project-wide random seed as consistently as possible.

    We centralize reproducibility here so every entrypoint can rely on the same
    behavior. The goal is not to guarantee perfect bitwise determinism for every
    external dependency, but to make repeated benchmark runs as stable as
    reasonably possible across Python, NumPy and PyTorch.
    """

    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
