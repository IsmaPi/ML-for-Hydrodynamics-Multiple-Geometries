"""Configuration dataclasses and YAML loading helpers."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "torchmd_et"      # "torchmd_gn" or "torchmd_et"
    input_dim: int = 3                  # forces only (3D)
    hidden_dim: int = 256
    num_layers: int = 4
    output_dim: int = 3                 # 3D displacement
    # TorchMD-NET specific
    cutoff_radius: float = 30.0
    num_rbf: int = 32
    rbf_type: str = "expnorm"
    trainable_rbf: bool = True
    num_heads: int = 8                  # ET only
    max_num_neighbors: int = 32
    distance_influence: str = "both"    # ET only: "keys", "values", or "both"
    neighbor_strategy: str = "brute"
    # TensorNet specific
    equivariance_invariance_group: str = "O(3)"
    # Mobility head (predicts alpha/beta per edge)
    mobility_hidden_dim: int = 64


@dataclass
class DataConfig:
    """Data generation and loading configuration."""
    geometries: List[str] = field(default_factory=lambda: ["nbody_open", "pse_periodic"])
    num_particles: List[int] = field(default_factory=lambda: [16, 32, 64])
    num_cloud_samples: int = 10000      # per geometry per N
    num_pair_samples: int = 2000        # per geometry
    pair_d_min: float = 2.0             # minimum pair distance (units of a)
    pair_d_max: float = 20.0            # maximum pair distance (units of a)
    force_scale: float = 1.0
    viscosity: float = 1.0 / (6.0 * math.pi)  # self_mobility = 1
    hydrodynamic_radius: float = 1.0
    output_dir: str = "data/raw"
    seed: int = 42
    # PSE-specific
    box_size: float = 32.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    max_epochs: int = 500
    gradient_clip: float = 0.5
    patience: int = 50
    seed: int = 42
    precision: str = "16-mixed"
    loss_type: str = "relative_mse"
    passive_weight: float = 5.0
    relative_loss_eps: float = 1e-6


def load_yaml(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_model_config(raw: dict) -> ModelConfig:
    """Build ModelConfig from a dict, ignoring unknown keys."""
    return ModelConfig(**{k: v for k, v in raw.items() if k in ModelConfig.__dataclass_fields__})


def build_data_config(raw: dict) -> DataConfig:
    """Build DataConfig from a dict, ignoring unknown keys."""
    return DataConfig(**{k: v for k, v in raw.items() if k in DataConfig.__dataclass_fields__})


def build_training_config(raw: dict) -> TrainingConfig:
    """Build TrainingConfig from a dict, ignoring unknown keys."""
    return TrainingConfig(**{k: v for k, v in raw.items() if k in TrainingConfig.__dataclass_fields__})


def resolve_config_path(value: str, config_type: str) -> str:
    """Resolve short config names to full paths.

    Examples:
        resolve_config_path("mixed_all", "data") -> "configs/data/mixed_all.yaml"
        resolve_config_path("configs/data/mixed_all.yaml", "data") -> unchanged
    """
    p = Path(value)
    if p.exists():
        return value
    candidate = Path("configs") / config_type / (p.stem + ".yaml")
    if candidate.exists():
        return str(candidate)
    return value
