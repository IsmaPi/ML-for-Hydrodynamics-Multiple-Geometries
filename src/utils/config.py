from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml

DEFAULT_NORMALIZER_PATH = "data/processed/normalizer.pt"


# ---------------------------------------------------------------------------
# Geometry configuration
# ---------------------------------------------------------------------------
@dataclass
class GeometryConfig:
    solver: str                        # "NBody", "DPStokes", "PSE"
    geometry: str                      # "open", "single_wall", "two_walls", "periodic"
    periodicity_x: str = "open"
    periodicity_y: str = "open"
    periodicity_z: str = "open"
    viscosity: float = 1.0
    hydrodynamic_radius: float = 1.0
    # Solver-specific parameters
    box_size_x: Optional[float] = None
    box_size_y: Optional[float] = None
    box_size_z: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None


# ---------------------------------------------------------------------------
# Data generation configuration
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    geometries: List[str] = field(default_factory=list)
    num_particles: List[int] = field(default_factory=lambda: [64])
    num_trajectories: int = 200
    trajectory_length: int = 50
    dt: float = 0.01
    force_scale: float = 1.0
    viscosity: float = 1.0
    hydrodynamic_radius: float = 1.0
    seed: int = 42
    output_dir: str = "data/raw"
    use_synthetic: bool = False


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    model_type: str = "torchmd_gn"     # "torchmd_gn" or "torchmd_et"
    input_dim: int = 9                 # [force(3), theta(6)]
    hidden_dim: int = 256
    num_layers: int = 4
    output_dim: int = 3                # 3D translational displacement
    layer_norm: bool = True
    dropout: float = 0.0
    # TorchMD-NET specific
    cutoff_radius: float = 30.0
    num_rbf: int = 32
    rbf_type: str = "expnorm"
    trainable_rbf: bool = True
    max_num_neighbors: int = 32
    num_heads: int = 8                 # for TorchMD_ET multi-head attention
    distance_influence: str = "both"   # for TorchMD_ET: "keys", "values", or "both"
    neighbor_strategy: str = "brute"   # "brute" (N≤128) or "cell" (N>1000)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    strategy: str = "mixed"            # "single", "mixed", "leave_one_out"
    leave_out_geometry: Optional[str] = None
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 200
    scheduler: str = "cosine"          # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    early_stopping_patience: int = 50
    seed: int = 42
    log_every: int = 10
    eval_every: int = 5
    checkpoint_every: int = 20
    log_backend: str = "tensorboard"   # "wandb" or "tensorboard"


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------
def resolve_config_path(value: str, config_type: str) -> str:
    """Resolve short config names to full paths.

    Accepts both full paths and short names. Short names are resolved to
    configs/<config_type>/<name>.yaml.

    Args:
        value: Config path or short name (e.g. "mixed_all" or "configs/data/mixed_all.yaml")
        config_type: One of "data", "model", "training"

    Examples:
        resolve_config_path("mixed_all", "data") -> "configs/data/mixed_all.yaml"
        resolve_config_path("configs/data/mixed_all.yaml", "data") -> "configs/data/mixed_all.yaml"
    """
    p = Path(value)
    if p.exists():
        return value
    candidate = Path("configs") / config_type / (p.stem + ".yaml")
    if candidate.exists():
        return str(candidate)
    return value


def load_yaml(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_configs(*dicts: dict) -> dict:
    """Shallow-merge multiple config dicts (last wins)."""
    merged: Dict[str, Any] = {}
    for d in dicts:
        merged.update(d)
    return merged


def build_data_config(raw: dict) -> DataConfig:
    return DataConfig(**{k: v for k, v in raw.items() if k in DataConfig.__dataclass_fields__})


def build_model_config(raw: dict) -> ModelConfig:
    return ModelConfig(**{k: v for k, v in raw.items() if k in ModelConfig.__dataclass_fields__})


def build_training_config(raw: dict) -> TrainingConfig:
    return TrainingConfig(**{k: v for k, v in raw.items() if k in TrainingConfig.__dataclass_fields__})
