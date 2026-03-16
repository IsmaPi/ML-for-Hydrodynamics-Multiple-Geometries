# ML for Hydrodynamics: Generalization Across Multiple Geometries

Applying Machine Learning to predict deterministic particle displacements in low-Reynolds-number hydrodynamics, comparing a **SchNet-based GN** (TorchMD_GN) and an **Equivariant Transformer** (TorchMD_ET) on their ability to generalize across boundary geometries.

Supervised by Prof. R. P. Pelaez (Universidad Autonoma de Madrid / IE University).

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate hydro-ml
```

### 2. Install libMobility

libMobility is required for data generation (provides ground-truth mobility solvers).

```bash
# Follow installation instructions from:
# https://github.com/stochasticHydroTools/libMobility
```

## Quick Start

### Run the full pipeline

```bash
conda activate hydro-ml
python run.py --stage all --model torchmd_et
```

This will: generate data (libMobility) -> train TorchMD_ET -> evaluate.

### Run with both models

```bash
python run.py --stage all --model both
```

## Step-by-Step Usage

### 1. Generate data

```bash
# All geometries (default)
python scripts/generate_data.py --config default

# Single geometry
python scripts/generate_data.py --config nbody_open
python scripts/generate_data.py --config pse_periodic
```

### 2. Train a model

Training uses PyTorch Lightning. Models are saved to `saved_models/`, logs to `logs/`.

```bash
# Train Equivariant Transformer
python scripts/train.py \
    --data-config default \
    --model-config torchmd_et

# Train SchNet (GN)
python scripts/train.py \
    --data-config default \
    --model-config torchmd_gn

# Custom run name
python scripts/train.py \
    --data-config default \
    --model-config torchmd_et \
    --run-name my_experiment
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs/
```

### 3. Evaluate

```bash
# Single-step accuracy
python scripts/evaluate.py \
    --checkpoint saved_models/torchmd_et/run_001/best.ckpt \
    --eval-mode single_step \
    --data-config default

# Two-particle distance sweep vs libMobility ground truth
python scripts/evaluate.py \
    --checkpoint saved_models/torchmd_et/run_001/best.ckpt \
    --eval-mode pair_sweep
```

## Supported Geometries

| Geometry | Solver | Periodicities (x, y, z) |
|---|---|---|
| `nbody_open` | NBody | open, open, open |
| `pse_periodic` | PSE | periodic, periodic, periodic |

## Configuration

All hyperparameters are in YAML config files under `configs/`:

- `configs/model/torchmd_gn.yaml`: hidden_dim, num_layers, cutoff_radius, num_rbf
- `configs/model/torchmd_et.yaml`: hidden_dim, num_layers, num_heads, distance_influence
- `configs/training/default.yaml`: learning_rate, batch_size, max_epochs, patience
- `configs/data/default.yaml`: geometries, num_particles, num_cloud_samples, num_pair_samples

## Project Structure

```
data/           # Data generation (libMobility) and PyTorch Dataset/DataModule
models/         # TorchMD_ET, TorchMD_GN, Lightning wrapper
evaluation/     # Single-step metrics and pair sweep validation
utils/          # Config dataclasses, seed utilities
scripts/        # CLI entry points (generate, train, evaluate)
configs/        # YAML configuration files
run.py          # Main pipeline: generate -> train -> evaluate
```

## License

GPL-3.0
