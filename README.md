# Codebase Technical Summary

## 1. Problem Statement & Physics

This project learns the **hydrodynamic mobility operator** M(X) for Stokes flow using graph neural networks. The governing equation is:

**U = M(X) · F**

where:
- **U** ∈ R^{N×3}: per-particle displacements (velocities in Stokes regime)
- **M(X)** ∈ R^{3N×3N}: configuration-dependent mobility tensor (depends on all particle positions X)
- **F** ∈ R^{N×3}: applied forces on each particle

The mobility tensor M decomposes into:
- **Self-mobility block**: M_ii = (1 / 6πηa) · I (identity, known analytically)
- **Pair interaction blocks**: M_ij for i≠j, described by the **Rotne-Prager-Yamakawa (RPY) tensor**:
  - M_ij = (1/8πηr) · [a(r)·I + b(r)·r̂⊗r̂]
  - a(r) and b(r) are scalar functions of inter-particle distance
  - The **r̂⊗r̂ term is anisotropic** — it couples displacement direction to the inter-particle axis

**Units**: viscosity η = 1/(6π), hydrodynamic radius a = 1.0, giving **self-mobility = 1** (displacement = force for isolated particles).

**Two geometries** are supported:
- **nbody_open**: Unbounded domain (free space), RPY tensor via libMobility's NBody solver
- **pse_periodic**: Fully periodic domain (Lx=Ly=Lz=32), Periodic Stokes Equations via libMobility's PSE solver with Ewald splitting (ψ=1.0)

---

## 2. Data Generation Pipeline

**File**: `data/generate.py` | **Solver**: `utils/solver.py`

Ground truth displacements are computed by calling `solver.Mdot(forces=F)` on initialized libMobility solvers, which compute the exact M·F product using either the RPY tensor (open) or Ewald summation (periodic).

### Sampling strategies

**Cloud samples** (`generate_cloud_samples`): Random N-particle configurations
- Particle counts: N ∈ {16, 32, 64}
- 5,000 samples per (geometry, N) combination = **30,000 cloud samples per geometry**
- Positions: uniform in [-10, 10]³ (open) or [0, 32]³ (periodic)
- Minimum particle separation enforced: 2.5a (rejection sampling)
- Forces: random unit direction (normalized Gaussian) × random magnitude ∈ [0, 1.0]

**Pair samples** (`generate_pair_samples`): Exactly 2 particles per sample
- **150,000 samples per geometry** (dominant data source)
- Inter-particle distance: uniform in [2a, 20a]
- Particle 1: random force (same distribution as clouds)
- Particle 2: **zero force** (passive — displacement comes entirely from hydrodynamic interaction)
- Open: particle 1 at origin, particle 2 at distance d in random direction
- Periodic: both centered in box to minimize periodic image effects

**Total dataset**: 2 geometries × (30,000 clouds + 150,000 pairs) = **330,000 samples**

### Data format

Each sample saved as `.npz` containing:
- `positions`: (N, 3) float32
- `forces`: (N, 3) float32
- `displacement`: (N, 3) float32 — ground truth M·F
- `geometry_id`: int32 — 0 for nbody_open, 1 for pse_periodic

**No normalization** is applied anywhere in the pipeline — all quantities are in natural Stokes units.

---

## 3. Dataset & Loading

**File**: `data/dataset.py`

### PyTorch Geometric Data construction

Each sample becomes a `torch_geometric.data.Data` object:
- `data.x`: (N, 4) — forces (3 columns) + geometry_id (1 column, broadcast to all particles)
- `data.pos`: (N, 3) — particle positions
- `data.y`: (N, 3) — target displacement
- No explicit edge_index — TorchMD-NET constructs the neighbor graph internally using cutoff radius

### Train/Val/Test split

- **70% train / 10% val / 20% test** (deterministic with seed=42)
- Random assignment (not stratified by geometry or particle count)
- Train: 230,999 samples | Val: 33,001 | Test: 66,000

### DataLoader

- Batch size: 64
- PyTorch Geometric batching: concatenates node features with `batch` assignment vector
- Train: shuffled | Val/Test: sequential
- num_workers: min(cpu_count-1, 8), pin_memory=True

---

## 4. Model Architectures

All three models follow the same **delta learning** pattern:

**output = self_mobility × F + learned_correction(X, F)**

The self-mobility skip connection provides the dominant isotropic baseline. The model only needs to learn the **interaction corrections** (the off-diagonal blocks of M).

All models use TorchMD-NET backbones with the default integer atom-type embedding **replaced** by a learned linear projection from 4D input features (forces + geometry_id) to hidden_dim.

### 4a. HydroTorchMD_GN (SchNet Baseline)

**File**: `models/torchmd_gn.py` | **Config**: `configs/model/torchmd_gn.yaml`

| Property | Value |
|----------|-------|
| Backbone | TorchMD_GN (SchNet) |
| Hidden dim | 128 |
| Layers | 3 |
| Parameters | ~228K |
| Message passing | **Isotropic scalar** — filters depend on distance only, not direction |
| Output | `output_head(scalar_features)` → (N, 3) |
| Vector path | **None** |
| Equivariance | **Not equivariant** — scalar messages lose directional information |

**Architecture**:
```
Input (N, 4) → Linear(4, 128) → SchNet backbone (3 layers, 128 channels, 32 expnorm RBFs)
                                    → scalar features (N, 128)
                                    → Linear(128, 128) → SiLU → Linear(128, 3) [zero-init]
                                    → displacement correction (N, 3)
Skip: self_mobility × forces[:, :3]
```

**Key limitation**: SchNet's continuous-filter convolutions are functions of scalar distance only. The output head maps invariant scalars to 3D vectors, which **cannot represent the anisotropic r̂⊗r̂ term** of the RPY tensor in a rotation-consistent manner. It can approximate this term for the training distribution but generalizes poorly under rotation.

### 4b. HydroTorchMD_ET (Equivariant Transformer)

**File**: `models/torchmd_et.py` | **Config**: `configs/model/torchmd_et.yaml`

| Property | Value |
|----------|-------|
| Backbone | TorchMD_ET (Equivariant Transformer) |
| Hidden dim | 128 |
| Layers | 2 |
| Attention heads | 4 |
| Parameters | ~415K |
| Message passing | **Equivariant** — attention uses both distance and direction |
| Output | `output_head(scalar_features)` + `vector_proj(vector_features)` |
| Vector path | **Yes** — projects backbone vector features (N, 3, 128) → (N, 3) |
| Equivariance | **SE(3) equivariant** via equivariant attention mechanism |

**Architecture**:
```
Input (N, 4) → Linear(4, 128) → ET backbone (2 layers, 4 heads, distance_influence="both")
                                    → scalar features (N, 128), vector features (N, 3, 128)

Scalar path: Linear(128, 128) → SiLU → Linear(128, 3) [zero-init]  → correction (N, 3)
Vector path: Linear(128, 1, bias=False) [zero-init]                  → correction (N, 3)

Skip: self_mobility × forces[:, :3]
Output = skip + scalar_correction + vector_correction
```

**Vector features**: The ET backbone maintains per-particle vector representations (N, 3, hidden_dim) through equivariant attention layers. These vectors transform correctly under rotation (R·v). The `vector_proj` layer (bias-free, zero-initialized) maps them to a 3D correction that is **equivariant by construction**.

### 4c. HydroTorchMD_TN (TensorNet)

**File**: `models/torchmd_tn.py` | **Config**: `configs/model/torchmd_tn.yaml`

| Property | Value |
|----------|-------|
| Backbone | TensorNet |
| Hidden dim | 128 |
| Layers | 2 |
| Parameters | ~763K |
| Message passing | **Tensor product interactions** (rank-0, rank-1, rank-2) |
| Output | `output_head(scalar_features)` + `vector_proj(antisymmetric_vectors)` |
| Vector path | **Yes** — extracted from antisymmetric part of internal 3×3 tensor |
| Equivariance | **O(3) equivariant** (rotations + reflections) |

**Architecture**:
```
Input (N, 4) → Linear(4, 128) → TensorNet backbone (2 layers, O(3), static_shapes=False)
                                    → scalar features (N, 128)
                                    → [hook captures] final tensor X (N, 3, 3, 128)

Scalar path: Linear(128, 128) → SiLU → Linear(128, 3) [zero-init]  → correction (N, 3)
Vector path: A = antisymmetric(X)  → Hodge dual → vec (N, 3, 128)
             Linear(128, 1, bias=False) [zero-init]                  → correction (N, 3)

Skip: self_mobility × forces[:, :3]
Output = skip + scalar_correction + vector_correction
```

**Tensor internals**: TensorNet maintains a per-particle 3×3×hidden_dim tensor throughout the network. This tensor decomposes into:
- **I** (rank-0, scalar): trace — invariant under rotations
- **A** (rank-1, antisymmetric): skew-symmetric part — encodes pseudovectors
- **S** (rank-2, symmetric traceless): encodes quadrupolar interactions

Standard TensorNet contracts all three to scalar norms for energy prediction. We intercept `A` via a forward hook on the last interaction layer and extract equivariant vectors using the Hodge dual: `vec = [A₁₂−A₂₁, A₂₀−A₀₂, A₀₁−A₁₀]`.

**Note on `static_shapes=False`**: Required because TensorNet's default embedding expects 1D integer atomic numbers, but we pass 2D continuous features. An explicit `q=torch.zeros(N)` is passed to avoid shape mismatches in the charge modulation.

---

## 5. Training Setup

**File**: `models/lightning_wrapper.py` | **Config**: `configs/training/default.yaml`

### Loss function

```python
loss = MSE = mean_over_particles[ sum_over_xyz( (pred - target)² ) ]
```

Current config: standard MSE with no passive weighting (passive_weight=1.0). A relative MSE option exists but is not currently active.

### Optimizer & scheduler

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-3)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=15, min_lr=1e-6)
- **Gradient clipping**: global norm ≤ 1.0

### Early stopping

- Monitors `val_loss`
- Patience: 80 epochs
- Saves best checkpoint only (by val_loss)

### Logged metrics

**Training**: `train_loss`, `train_mse`, `train_correction_rms` (learned correction magnitude), `lr`

**Validation**: `val_loss`, `val_mse`, `val_baseline_mse` (self-mobility only), `val_improvement` (= 1 − pred_mse/baseline_mse), `val_rel_error`, `val_mae_{x,y,z}`, `val_correction_rms`

**Test**: `test_mse`, `test_mae`, `test_rel_error`

The `val_improvement` metric is key: it measures how much the model improves over the analytical self-mobility baseline. A value of 0.78 means the model reduces MSE by 78% compared to ignoring interactions entirely.

---

## 6. Evaluation Pipeline

**Files**: `evaluation/single_step.py`, `evaluation/pair_sweep.py`, `run.py`

### Single-step evaluation

Computes MSE, MAE, and relative error on the full validation set. Provides aggregate model accuracy.

### Pair sweep evaluation

The core diagnostic tool. Places two particles at varying distances d ∈ [2.5a, 15a]:
- Particle 1: force = [1, 0, 0]
- Particle 2: force = [0, 0, 0] (passive)

Compares model predictions vs analytical solver for both particles across all 3 displacement components. Generates 4 plots:
1. **Particle 1 displacement** (force-applied): x-component should approach self-mobility (=1) at large d
2. **Particle 2 displacement** (bystander): reveals learned hydrodynamic coupling
3. **Both particles** in 2×3 grid
4. **Error** (absolute + relative) vs distance

---

## 7. Current Results & Potential Issues

### Latest results (with vector paths + geometry tagging)

| Model | Params | val_improvement | test_mse | test_rel_error |
|-------|--------|----------------|----------|---------------|
| GN | 228K | 78.3% | 0.0036 | 0.417 |
| ET | 415K | 78.4% | 0.0035 | 0.411 |
| TN | 763K | 76.2% | 0.0035 | 0.488 |

### Identified issues and potential causes

**1. All models plateau at ~78% improvement / ~41% relative error**
- The scalar `output_head` maps invariant features to 3D, which cannot represent direction-dependent corrections in an equivariant way
- The vector correction paths (ET/TN) are zero-initialized and may need more training signal or larger models to activate
- With only 2 layers in ET/TN, vector features may not have enough depth to develop rich representations

**2. Training data composition**
- 150,000 pair samples vs 30,000 cloud samples per geometry — dataset is dominated by 2-particle configurations
- Pair samples always have particle 2 with zero force — model may not learn symmetric two-body interactions well
- No intermediate particle counts (N=8, N=128) — generalization across system sizes is untested

**3. Mixed geometries without clear separation**
- Geometry ID is appended as a single scalar feature — the model must learn to route through different physics based on one number
- Open (RPY) vs periodic (PSE) have fundamentally different long-range behavior; a single model may struggle to represent both
- The model might benefit from geometry-specific output heads or separate training

**4. Loss function**
- Standard MSE weights all particles equally, but pair samples have one active + one passive particle
- The passive particle's displacement is much smaller (pure interaction), so its signal is dominated by the active particle's self-mobility term
- Relative MSE or increased passive_weight could help learn interaction effects

**5. Cutoff radius vs box size**
- Cutoff = 30.0 with box size = 32.0 means periodic interactions beyond the cutoff are missed
- For the open geometry, particles at spread=10 can be up to 20√3 ≈ 34.6 apart — near the cutoff

**6. No data augmentation**
- Rotational augmentation would reinforce equivariance for the GN (which is not equivariant by architecture)
- Force magnitude augmentation could help generalization

---

## 8. File Map

| File | Purpose |
|------|---------|
| `run.py` | Main entry: generate → train → evaluate pipeline |
| `data/generate.py` | Data generation via libMobility solvers |
| `data/dataset.py` | PyTorch Geometric dataset + datamodule |
| `utils/solver.py` | Solver factory + SolverCallable wrapper |
| `utils/config.py` | Dataclass configs + YAML loading |
| `models/torchmd_gn.py` | SchNet baseline (scalar-only) |
| `models/torchmd_et.py` | Equivariant Transformer (scalar + vector) |
| `models/torchmd_tn.py` | TensorNet (scalar + antisymmetric vector) |
| `models/lightning_wrapper.py` | Training loop, loss, metrics, optimizer |
| `evaluation/single_step.py` | Aggregate MSE/MAE/RelErr on validation set |
| `evaluation/pair_sweep.py` | Two-particle diagnostic plots |
| `configs/model/*.yaml` | Per-model architecture configs |
| `configs/data/*.yaml` | Data generation configs |
| `configs/training/default.yaml` | Training hyperparameters |
