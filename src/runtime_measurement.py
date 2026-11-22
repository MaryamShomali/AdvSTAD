"""Consistent CPU/CUDA timing boundaries without changing optimization."""

from time import perf_counter

import torch


def synchronized_time(device):
    """Drain queued CUDA work, then read a monotonic high-resolution clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter()
