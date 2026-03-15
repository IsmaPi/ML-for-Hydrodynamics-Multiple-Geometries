"""Single-step evaluation metrics.

Computes MSE, MAE, and relative error between predicted and true displacements.
No denormalization needed — model outputs physical units directly.
"""

import torch
import numpy as np
from torch_geometric.loader import DataLoader


@torch.no_grad()
def evaluate_single_step(model, dataset, device, batch_size=64):
    """Evaluate single-step prediction accuracy.

    Args:
        model: trained model (in eval mode)
        dataset: HydrodynamicsDataset or Subset
        device: torch device
        batch_size: evaluation batch size

    Returns:
        dict with mse, mae, relative_error
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_errors_sq = []
    all_errors_abs = []
    all_rel_errors = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        diff = pred - batch.y

        # Per-particle squared error
        errors_sq = (diff ** 2).sum(dim=-1)  # (N_total,)
        all_errors_sq.append(errors_sq.cpu())

        # Per-particle absolute error
        errors_abs = diff.abs().sum(dim=-1)  # (N_total,)
        all_errors_abs.append(errors_abs.cpu())

        # Per-particle relative error
        true_norm = batch.y.norm(dim=-1).clamp(min=1e-12)
        rel_error = diff.norm(dim=-1) / true_norm
        all_rel_errors.append(rel_error.cpu())

    all_errors_sq = torch.cat(all_errors_sq)
    all_errors_abs = torch.cat(all_errors_abs)
    all_rel_errors = torch.cat(all_rel_errors)

    return {
        "mse": all_errors_sq.mean().item(),
        "mae": all_errors_abs.mean().item(),
        "relative_error": all_rel_errors.mean().item(),
        "num_samples": len(all_errors_sq),
    }
