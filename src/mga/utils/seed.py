"""Random seed utilities for reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch (CPU and CUDA).
    Also configures PyTorch for deterministic behavior.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
