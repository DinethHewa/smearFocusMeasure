"""Global reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np


def set_global_seed(seed: int, *, deterministic_hash_seed: bool = True) -> Dict[str, int]:
    """Set the common random seeds used by the scaffold."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass

    return {
        "python_random_seed": seed,
        "numpy_seed": seed,
        "pythonhashseed": seed,
    }


__all__ = ["set_global_seed"]

