# ML for Hydrodynamics: Generalization Across Multiple Geometries

Applying Machine Learning to predict deterministic particle displacements in low-Reynolds-number hydrodynamics, comparing a **Graph Neural Network (GNN)** and a **Set Transformer** on their ability to generalize across multiple boundary geometries.

Supervised by Prof. R. P. Pelaez (Universidad Autonoma de Madrid / IE University).

## Setup

### 1. Create the conda environment

```bash
conda create -n hydro-ml python=3.11 numpy scipy pyyaml matplotlib -y --channel conda-forge
conda activate hydro-ml
pip install torch torch_geometric torch-cluster torch-scatter
```

### 2. Install libMobility for real data generation

When a CUDA-enabled GPU is available:

```bash
# Follow libMobility installation instructions from:
# https://github.com/stochasticHydroTools/libMobility
```

Without libMobility, the pipeline uses a synthetic Oseen-tensor approximation for development and testing.

## Quick Start

### Run the full pipeline (synthetic data, no GPU needed)

```bash
conda activate hydro-ml
python run.py --synthetic --data-config configs/data/nbody_open.yaml
```

This will: generate synthetic data -> train a GNN -> evaluate it.

### Run with both models

```bash
python run.py --synthetic --model both --data-config configs/data/nbody_open.yaml
```

## Step-by-Step Usage

### 1. Generate data

```bash
# Single geometry with synthetic data
python scripts/generate_data.py --config configs/data/nbody_open.yaml --synthetic

# All geometries with synthetic data
python scripts/generate_data.py --config configs/data/mixed_all.yaml --synthetic

# Specific particle count
python scripts/generate_data.py --config configs/data/nbody_open.yaml --synthetic --num-particles 32

# With libMobility (requires CUDA GPU)
python scripts/generate_data.py --config configs/data/mixed_all.yaml
```

### 2. Train a model

Training automatically saves:
- **Model weights** to `saved_models/<run_name>/best.pt`
- **Training history** (per-epoch losses) to `results/<run_name>/history.csv`
- **Summary metrics** to `results/<run_name>/summary.json`

```bash
# Train GNN on a single geometry
python scripts/train.py \
    --data-config configs/data/nbody_open.yaml \
    --model-config configs/model/gnn.yaml \
    --train-config configs/training/single_geometry.yaml

# Train Set Transformer on mixed geometries
python scripts/train.py \
    --data-config configs/data/mixed_all.yaml \
    --model-config configs/model/set_transformer.yaml \
    --train-config configs/training/mixed_geometry.yaml

# Leave-one-out generalization experiment
python scripts/train.py \
    --data-config configs/data/mixed_all.yaml \
    --model-config configs/model/gnn.yaml \
    --train-config configs/training/leave_one_out.yaml

# Custom run name
python scripts/train.py \
    --data-config configs/data/nbody_open.yaml \
    --model-config configs/model/gnn.yaml \
    --train-config configs/training/single_geometry.yaml \
    --run-name my_experiment
```

### 3. Evaluate (using saved models, no retraining needed)

Evaluation loads models from `saved_models/` and writes results to `results/`.

```bash
# Single-step accuracy per geometry
python scripts/evaluate.py \
    --data-config configs/data/nbody_open.yaml \
    --model-config configs/model/gnn.yaml \
    --checkpoint saved_models/gnn_single_geometry/best.pt \
    --eval-mode single_step

# Cross-geometry generalization report
python scripts/evaluate.py \
    --data-config configs/data/mixed_all.yaml \
    --model-config configs/model/gnn.yaml \
    --checkpoint saved_models/gnn_mixed_geometry/best.pt \
    --eval-mode generalization

# Scaling: inference time/memory vs number of particles
python scripts/evaluate.py \
    --model-config configs/model/gnn.yaml \
    --checkpoint saved_models/gnn_single_geometry/best.pt \
    --eval-mode scaling

# Custom output path
python scripts/evaluate.py \
    --data-config configs/data/mixed_all.yaml \
    --model-config configs/model/gnn.yaml \
    --checkpoint saved_models/gnn_mixed_geometry/best.pt \
    --eval-mode generalization \
    --output my_results.json
```

## Project Structure

```
configs/                    # YAML configuration files
  data/                     #   Data generation configs per geometry
  model/                    #   Model architecture configs (GNN, Set Transformer)
  training/                 #   Training strategy configs (single, mixed, leave-one-out)
src/
  data/
    generate.py             # Trajectory generation (libMobility + synthetic fallback)
    geometry.py             # Geometry/solver registry and theta vector construction
    dataset.py              # PyTorch Dataset with PyG Data objects
    normalization.py        # Per-feature mean/std normalization
    graph_construction.py   # k-NN and radius graph builders
  models/
    gnn.py                  # Message-passing GNN (Eq. 10-12)
    set_transformer.py      # Set Transformer with SAB/ISAB (Eq. 13)
    common.py               # Shared MLP blocks
  training/
    trainer.py              # Training loop with per-geometry validation
    losses.py               # Displacement MSE loss (Eq. 14)
  evaluation/
    single_step.py          # Single-step accuracy metrics
    rollout.py              # Autoregressive rollout evaluation
    scaling.py              # Inference time/memory profiling
    generalization.py       # Cross-geometry generalization analysis
  utils/
    config.py               # YAML config loading + dataclasses
    seed.py                 # Reproducibility utilities
    logging.py              # wandb/tensorboard wrapper
scripts/
  generate_data.py          # CLI for data generation
  train.py                  # CLI for training
  evaluate.py               # CLI for evaluation
run.py                      # Main script to run the full pipeline
saved_models/               # Trained model weights (by run name)
  gnn_single_geometry/
    best.pt                 #   Best model checkpoint
  set_transformer_mixed_geometry/
    best.pt
results/                    # Training metrics and evaluation results (by run name)
  gnn_single_geometry/
    history.csv             #   Per-epoch train/val losses
    summary.json            #   Final training summary
    eval_single_step.json   #   Single-step evaluation results
    eval_scaling.json       #   Scaling profiling results
```

## Supported Geometries

| Geometry | Solver | Periodicities (x, y, z) |
|---|---|---|
| `nbody_open` | NBody | open, open, open |
| `nbody_single_wall` | NBody | open, open, single_wall |
| `dpstokes_two_walls` | DPStokes | periodic, periodic, two_walls |
| `pse_periodic` | PSE | periodic, periodic, periodic |

## Training Strategies

- **Single geometry**: Train and evaluate on one geometry (baseline).
- **Mixed geometry**: Train on all geometries combined, evaluate on each separately.
- **Leave-one-out**: Train on 3 geometries, test generalization on the held-out 4th.

## Configuration

All hyperparameters are in YAML config files under `configs/`. Key parameters:

- `configs/model/gnn.yaml`: hidden_dim, num_layers, graph_method, k_neighbors
- `configs/model/set_transformer.yaml`: hidden_dim, num_layers, num_heads, num_inducing_points
- `configs/training/*.yaml`: learning_rate, batch_size, num_epochs, strategy

## License

GPL-3.0
