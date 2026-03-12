"""Cross-geometry generalization evaluation.

Computes a generalization matrix: how well a model trained on some
geometries performs when tested on each geometry individually.
"""

import logging

import numpy as np
import torch
from typing import Dict, Optional

log = logging.getLogger(__name__)

from .single_step import evaluate_single_step
from ..data.normalization import FeatureNormalizer


def evaluate_cross_geometry(
    model: torch.nn.Module,
    geometry_datasets: Dict[str, object],
    normalizer: Optional[FeatureNormalizer],
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Evaluate model on each geometry separately.

    Args:
        geometry_datasets: {geometry_key: dataset} for each geometry to test.
        normalizer: fitted normalizer for denormalization.

    Returns:
        results: {geometry_key: {mse, mae, relative_error}, ...,
                  "generalization_summary": {...}}
    """
    results = {}
    mse_values = {}

    for geo_name, dataset in geometry_datasets.items():
        if len(dataset) == 0:
            continue
        metrics = evaluate_single_step(model, dataset, normalizer, device, batch_size)
        results[geo_name] = metrics
        mse_values[geo_name] = metrics["mse"]

    # Compute summary statistics
    if mse_values:
        all_mse = list(mse_values.values())
        results["generalization_summary"] = {
            "avg_mse": np.mean(all_mse),
            "std_mse": np.std(all_mse),
            "max_mse": np.max(all_mse),
            "min_mse": np.min(all_mse),
            "worst_geometry": max(mse_values, key=mse_values.get),
            "best_geometry": min(mse_values, key=mse_values.get),
        }

    return results


def print_generalization_report(results: dict):
    """Log the generalization evaluation results."""
    log.info("\n%s\nCross-Geometry Generalization Report\n%s", "=" * 60, "=" * 60)

    for geo, metrics in results.items():
        if geo == "generalization_summary":
            continue
        log.info("  %s:", geo)
        log.info("    MSE:            %.6e", metrics["mse"])
        log.info("    MAE:            %.6e", metrics["mae"])
        log.info("    Relative Error: %.4f", metrics["relative_error"])

    if "generalization_summary" in results:
        summary = results["generalization_summary"]
        log.info("  Summary:")
        log.info("    Avg MSE:  %.6e +/- %.6e", summary["avg_mse"], summary["std_mse"])
        log.info("    Best:     %s (%.6e)", summary["best_geometry"], summary["min_mse"])
        log.info("    Worst:    %s (%.6e)", summary["worst_geometry"], summary["max_mse"])
    log.info("=" * 60)
