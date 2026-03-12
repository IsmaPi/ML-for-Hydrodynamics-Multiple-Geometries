"""Single-step evaluation metrics on held-out data."""

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from typing import Optional

from ..data.normalization import FeatureNormalizer


@torch.no_grad()
def evaluate_single_step(
    model: torch.nn.Module,
    dataset,
    normalizer: Optional[FeatureNormalizer],
    device: torch.device,
    batch_size: int = 32,
    return_per_sample: bool = False,
) -> dict:
    """Evaluate single-step prediction accuracy.

    Metrics:
      - mse: mean squared error per particle per dimension
      - mae: mean absolute error
      - relative_error: ||pred - true||_2 / ||true||_2

    Returns: dict of metric_name -> value.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_pred = []
    all_true = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)

        # Denormalize
        if normalizer:
            pred = normalizer.inverse_transform(pred.cpu(), "displacements")
            true = normalizer.inverse_transform(batch.y.cpu(), "displacements")
        else:
            pred = pred.cpu()
            true = batch.y.cpu()

        all_pred.append(pred)
        all_true.append(true)

    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)

    mse = ((all_pred - all_true) ** 2).mean().item()
    mae = (all_pred - all_true).abs().mean().item()

    true_norm = torch.norm(all_true, dim=-1)
    err_norm = torch.norm(all_pred - all_true, dim=-1)
    relative_error = (err_norm / (true_norm + 1e-8)).mean().item()

    result = {"mse": mse, "mae": mae, "relative_error": relative_error}

    if return_per_sample:
        per_sample_mse = ((all_pred - all_true) ** 2).mean(dim=-1)
        per_sample_rel = err_norm / (true_norm + 1e-8)
        result["per_sample_mse"] = per_sample_mse.numpy().tolist()
        result["per_sample_rel_error"] = per_sample_rel.numpy().tolist()

    return result
