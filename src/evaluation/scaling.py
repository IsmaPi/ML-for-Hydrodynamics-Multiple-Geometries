"""Scaling evaluation: inference time and memory vs number of particles N."""

import time
import torch
import numpy as np
from torch_geometric.data import Data
from typing import List

from ..data.graph_construction import build_graph


def create_synthetic_input(
    N: int, input_dim: int, device: torch.device,
    graph_method: str = "knn", k: int = 16,
) -> Data:
    """Create a synthetic PyG Data sample for profiling."""
    pos = torch.randn(N, 3, device=device)
    x = torch.randn(N, input_dim, device=device)
    edge_index, edge_attr = build_graph(pos.cpu(), graph_method, k=k)

    data = Data(
        x=x,
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device),
        pos=pos,
    )
    data.batch = torch.zeros(N, dtype=torch.long, device=device)
    return data


@torch.no_grad()
def profile_scaling(
    model: torch.nn.Module,
    input_dim: int,
    device: torch.device,
    N_values: List[int] = None,
    graph_method: str = "knn",
    k: int = 16,
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
        data = create_synthetic_input(N, input_dim, device, graph_method, k)

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
