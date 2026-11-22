"""Seed the existing RNGs without changing initialization or backend policies."""

import os
import random
from numbers import Integral

import numpy as np
import torch

from src.study_protocol import DEFAULT_SEED


_seeded_with = None


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    return int(seed)


def seed_everything(seed=DEFAULT_SEED):
    """Keep the original torch.manual_seed call position and seed all libraries.

    torch.manual_seed also seeds all CUDA generators (lazily, without allocating
    model tensors). No deterministic/TF32/algorithm settings are changed here.
    """
    global _seeded_with
    seed = validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _seeded_with = seed


def seeding_metadata():
    return {
        "torch_seed": torch.initial_seed(),
        "numpy_seed": _seeded_with,
        "python_seed": _seeded_with,
        "cuda_seed": torch.initial_seed() if torch.cuda.is_available() else None,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "deterministic_algorithms": getattr(torch, "are_deterministic_algorithms_enabled", lambda: False)(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "checkpoint_rng_state_restored": False,
        "seed_policy": "python_numpy_torch_cuda_v1",
    }
