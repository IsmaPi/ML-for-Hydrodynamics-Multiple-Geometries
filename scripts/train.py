#!/usr/bin/env python3
"""CLI entry point for training.

Usage:
    python scripts/train.py \
        --data-config configs/data/mixed_all.yaml \
        --model-config configs/model/gnn.yaml \
        --train-config configs/training/mixed_geometry.yaml

    python scripts/train.py \
        --data-config configs/data/nbody_open.yaml \
        --model-config configs/model/set_transformer.yaml \
        --train-config configs/training/single_geometry.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import (
    load_yaml, merge_configs,
    build_data_config, build_model_config, build_training_config,
)
from src.utils.seed import set_seed
from src.utils.logging import Logger
from src.data.dataset import (
    HydrodynamicsDataset, compute_normalization_stats, split_dataset,
)
from src.models.gnn import HydroGNN
from src.models.set_transformer import HydroSetTransformer
from src.models.torchmd_gn import HydroTorchMD_GN
from src.models.torchmd_et import HydroTorchMD_ET
from src.training.trainer import Trainer

TORCHMD_MODELS = {"torchmd_gn", "torchmd_et"}


def build_datasets(data_cfg, model_cfg, train_cfg):
    """Build train/val datasets according to training strategy.

    Returns:
        train_dataset: combined training dataset
        val_datasets: dict of {geometry_key: dataset} for per-geo validation
    """
    all_geos = data_cfg.geometries
    particle_counts = data_cfg.num_particles

    if train_cfg.strategy == "leave_one_out" and train_cfg.leave_out_geometry:
        train_geos = [g for g in all_geos if g != train_cfg.leave_out_geometry]
        held_out_geos = [train_cfg.leave_out_geometry]
    else:
        train_geos = all_geos
        held_out_geos = []

    use_torchmd = model_cfg.model_type in TORCHMD_MODELS

    # Load full dataset for training geometries
    full_dataset = HydrodynamicsDataset(
        data_dir=data_cfg.output_dir,
        geometry_keys=train_geos,
        particle_counts=particle_counts,
    )

    # Compute normalization stats on training data
    normalizer = compute_normalization_stats(full_dataset)
    normalizer.save("data/processed/normalizer.pt")

    # Apply normalizer; precompute Data objects (skipped for torchMD — built on-the-fly)
    full_dataset.normalizer = normalizer
    if not use_torchmd:
        full_dataset.precompute_graphs()
    train_dataset, val_in_dist = split_dataset(full_dataset, train_ratio=0.8, seed=train_cfg.seed)

    # Build per-geometry validation datasets
    val_datasets = {}

    # In-distribution validation: split by geometry
    for geo in train_geos:
        geo_dataset = HydrodynamicsDataset(
            data_dir=data_cfg.output_dir,
            geometry_keys=[geo],
            particle_counts=particle_counts,
            normalizer=normalizer,
        )
        if not use_torchmd:
            geo_dataset.precompute_graphs()
        _, val_geo = split_dataset(geo_dataset, train_ratio=0.8, seed=train_cfg.seed)
        if len(val_geo) > 0:
            val_datasets[geo] = val_geo

    # Out-of-distribution (held-out geometry)
    for geo in held_out_geos:
        held_ds = HydrodynamicsDataset(
            data_dir=data_cfg.output_dir,
            geometry_keys=[geo],
            particle_counts=particle_counts,
            normalizer=normalizer,
        )
        if not use_torchmd:
            held_ds.precompute_graphs()
        if len(held_ds) > 0:
            val_datasets[f"{geo}_OOD"] = held_ds

    return train_dataset, val_datasets


def build_model(model_cfg):
    """Instantiate model from config."""
    if model_cfg.model_type == "gnn":
        return HydroGNN(model_cfg)
    elif model_cfg.model_type == "set_transformer":
        return HydroSetTransformer(model_cfg)
    elif model_cfg.model_type == "torchmd_gn":
        return HydroTorchMD_GN(model_cfg)
    elif model_cfg.model_type == "torchmd_et":
        return HydroTorchMD_ET(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_cfg.model_type}")


def make_run_name(model_cfg, train_cfg) -> str:
    """Generate a descriptive run name: {model_type}_{strategy}[_{leave_out}]."""
    parts = [model_cfg.model_type, train_cfg.strategy]
    if train_cfg.strategy == "leave_one_out" and train_cfg.leave_out_geometry:
        parts.append(f"lo_{train_cfg.leave_out_geometry}")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Train hydrodynamics ML model")
    parser.add_argument("--data-config", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--train-config", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run name (default: auto-generated from model+strategy)")
    args = parser.parse_args()

    raw = merge_configs(
        load_yaml(args.data_config),
        load_yaml(args.model_config),
        load_yaml(args.train_config),
    )
    data_cfg = build_data_config(raw)
    model_cfg = build_model_config(raw)
    train_cfg = build_training_config(raw)

    run_name = args.run_name or make_run_name(model_cfg, train_cfg)

    set_seed(train_cfg.seed)

    # Ensure processed dir exists
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print(f"Run: {run_name}")
    print(f"Model: {model_cfg.model_type} | Strategy: {train_cfg.strategy}")
    print(f"Geometries: {data_cfg.geometries} | N: {data_cfg.num_particles}")

    train_dataset, val_datasets = build_datasets(data_cfg, model_cfg, train_cfg)
    print(f"Train samples: {len(train_dataset)} | Val sets: {list(val_datasets.keys())}")

    model = build_model(model_cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    logger = Logger(train_cfg, backend=train_cfg.log_backend)
    trainer = Trainer(model, train_dataset, val_datasets, train_cfg, logger, run_name=run_name)
    trainer.fit()
    logger.finish()

    print(f"\nModel:   saved_models/{run_name}/best.pt")
    print(f"Metrics: results/{run_name}/")


if __name__ == "__main__":
    main()
