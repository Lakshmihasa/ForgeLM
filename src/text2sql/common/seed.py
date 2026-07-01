"""Global seeding for reproducibility.

Seeds Python's `random`, NumPy, and Torch (CPU + all CUDA devices) from one call,
so a run is repeatable. numpy/torch are imported lazily, so this stays importable
in environments (e.g. lightweight tests) that don't have them.

`deterministic=True` additionally requests deterministic algorithms. It's OFF by
default because full GPU determinism has a real speed cost and some ops lack a
deterministic implementation — `warn_only=True` degrades gracefully rather than
crashing when that happens. Use it when you need bit-for-bit reproduction, not for
normal training.
"""

from __future__ import annotations

import os
import random

__all__ = ["set_seed", "seed_worker"]


def set_seed(seed: int, *, deterministic: bool = False) -> int:
    """Seed all RNGs. Returns the seed for convenience/logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            # Required by some CUDA deterministic kernels.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive each worker's seed from torch's initial
    seed so shuffling/augmentation is reproducible across workers.

    Usage: DataLoader(..., worker_init_fn=seed_worker, generator=g)
    """
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass