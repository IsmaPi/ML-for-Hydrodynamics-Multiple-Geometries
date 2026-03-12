"""Scaling evaluation: inference time and memory vs number of particles N."""

import time
import torch
import numpy as np
from torch_geometric.data import Data
from typing import List


TORCHMD_MODELS = {"torchmd_gn", "torchmd_et"}


def create_synthetic_input(
    N: int, input_dim: int, device: torch.device,
    model_type: str = "torchmd_gn",
    graph_method: str = "knn", k: int = 16,
    cutoff_radius: float = 30.0,
) -> Data:
    """Create a synthetic PyG Data sample for profiling.

    Positions are spread over a box that scales with N so the average
    neighbour count stays roughly constant and does not exceed
    max_num_neighbors for TorchMD-NET models.
    """
    import math
    # Scale box so average neighbor count stays below max_num_neighbors.
    # neighbors ≈ n * (4/3)π * r_cut³ → V = N * (4/3)π * r_cut³ / target
    target_neighbors = min(N - 1, 64)  # can't have more neighbors than particles
    volume = N * (4.0 / 3.0) * math.pi * cutoff_radius**3 / max(target_neighbors, 1)
    box_side = volume ** (1.0 / 3.0)
    pos = torch.rand(N, 3, device=device) * box_side
    x = torch.randn(N, input_dim, device=device)

    data = Data(x=x, pos=pos)
    data.batch = torch.zeros(N, dtype=torch.long, device=device)
    return data


@torch.no_grad()
def profile_scaling(
    model: torch.nn.Module,
    input_dim: int,
    device: torch.device,
    N_values: List[int] = None,
    model_type: str = "torchmd_gn",
    graph_method: str = "knn",
    k: int = 16,
    cutoff_radius: float = 30.0,
    warmup_runs: int = 5,
    timed_runs: int = 50,
) -> dict:
    """Measure inference time and peak GPU memory vs N.

    Returns: dict of N -> {"time_ms": float, "memory_mb": float}
    """
    if N_values is None:
        N_values = [16, 32, 64, 128, 256]

    model.eval()
    results = {}

    for N in N_values:
        data = create_synthetic_input(N, input_dim, device, model_type, graph_method, k, cutoff_radius)

        # Warmup
        for _ in range(warmup_runs):
            _ = model(data)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        for _ in range(timed_runs):
            _ = model(data)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) / timed_runs * 1000

        if device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        else:
            peak_mem_mb = 0.0

        results[N] = {"time_ms": elapsed_ms, "memory_mb": peak_mem_mb}
        print(f"  N={N}: {elapsed_ms:.2f} ms, {peak_mem_mb:.1f} MB")

    return results
