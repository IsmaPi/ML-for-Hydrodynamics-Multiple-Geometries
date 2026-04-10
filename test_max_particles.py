#!/usr/bin/env python3
"""Test model generalization with increasing particle counts at constant density.

Scales the box/spread with N so neighbor count stays constant, matching
training conditions. Reports relative error and cosine similarity per N.
"""

import math
import numpy as np
import torch
from torch_geometric.data import Data

from models.lightning_wrapper import HydroLitModule
from data.generate import _sample_positions, _sample_forces, create_solver, GEOMETRY_IDS
from utils.config import DataConfig


# Training densities (particles per unit volume)
TRAINING_DENSITY = {
    "nbody_open":    64.0 / (20.0 ** 3),    # 0.008
    "pse_periodic":  64.0 / (32.0 ** 3),    # ~0.00195
}


def find_latest_checkpoint(model_type="torchmd_tn"):
    from pathlib import Path
    parent = Path("saved_models") / model_type
    runs = sorted(parent.glob("run_*/best.ckpt"))
    return str(runs[-1])


def load_model_with_max_neighbors(ckpt_path, max_num_neighbors, device):
    """Load checkpoint, override max_num_neighbors for larger systems."""
    lit_model = HydroLitModule.load_from_checkpoint(ckpt_path)

    from torchmdnet.models.utils import OptimizedDistance
    model = lit_model.model
    old_dist = model.backbone.distance
    model.backbone.distance = OptimizedDistance(
        cutoff_lower=old_dist.cutoff_lower,
        cutoff_upper=old_dist.cutoff_upper,
        max_num_pairs=-max_num_neighbors,
        return_vecs=True,
        loop=True,
        long_edge_index=True,
    )

    # Re-register edge capture hook on the new distance module
    model._edge_cache = {}
    def _capture_edges(module, input, output):
        model._edge_cache["edge_index"] = output[0]
        model._edge_cache["edge_weight"] = output[1]
        model._edge_cache["edge_vec"] = output[2]
    model.backbone.distance.register_forward_hook(_capture_edges)

    lit_model.eval()
    lit_model.to(device)
    return lit_model.model


def compute_box_for_density(geometry, N, cutoff):
    """Compute box size / spread that maintains training density for N particles.

    Also returns the expected number of neighbors per particle.
    """
    rho = TRAINING_DENSITY[geometry]
    volume = N / rho

    if geometry == "pse_periodic":
        box_size = volume ** (1.0 / 3.0)
        spread = None
    else:  # nbody_open
        # spread is half the box side: positions in [-spread, spread]^3
        side = volume ** (1.0 / 3.0)
        spread = side / 2.0
        box_size = 32.0  # not used for open geometry

    # Expected neighbors within cutoff
    expected_neighbors = rho * (4.0 / 3.0) * math.pi * cutoff ** 3
    return box_size, spread, expected_neighbors


def make_sample(geometry, N, device, viscosity, a, box_size, spread, seed=999):
    """Generate one sample with N particles and compute ground truth."""
    rng = np.random.default_rng(seed)
    positions = _sample_positions(N, geometry, rng, a=a, box_size=box_size,
                                  spread=spread if spread else 10.0)
    forces = _sample_forces(N, 1.0, rng)

    solver = create_solver(geometry, viscosity, a, box_size)
    solver.setPositions(positions)
    displacement, _ = solver.Mdot(forces=forces)
    displacement = np.array(displacement, dtype=np.float32).reshape(N, 3)
    solver.clean()

    geo_id = GEOMETRY_IDS[geometry]
    L = box_size if geometry == "pse_periodic" else 1e6

    data = Data(
        x=torch.ones(N, dtype=torch.long, device=device),
        pos=torch.tensor(positions, dtype=torch.float32, device=device),
        forces=torch.tensor(forces, dtype=torch.float32, device=device),
        geometry_id=torch.tensor(float(geo_id), device=device),
        box_vecs=torch.tensor(
            [[L, 0, 0], [0, L, 0], [0, 0, L]], dtype=torch.float32, device=device
        ),
    )
    data.batch = torch.zeros(N, dtype=torch.long, device=device)
    data.num_nodes = N

    target = torch.tensor(displacement, dtype=torch.float32, device=device)
    return data, target


def compute_metrics(pred, target):
    mse = (pred - target).pow(2).mean().item()
    mae = (pred - target).abs().mean().item()
    target_norm = target.norm(dim=-1).clamp(min=1e-6)
    rel_err = ((pred - target).norm(dim=-1) / target_norm).mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
    return {"mse": mse, "mae": mae, "rel_error": rel_err, "cosine_sim": cos_sim}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = find_latest_checkpoint("torchmd_tn")
    print(f"Checkpoint: {ckpt}")

    data_cfg = DataConfig()
    cutoff = 30.0  # from model config

    geometries = ["nbody_open", "pse_periodic"]
    particle_counts = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    # Expected neighbors at training density (same for all N since density is constant)
    for geo in geometries:
        _, _, exp_n = compute_box_for_density(geo, 64, cutoff)
        print(f"  {geo}: expected ~{exp_n:.0f} neighbors at training density")

    # At training density with cutoff=30, expected neighbors are ~905 (nbody)
    # and ~221 (PSE).  We need max_neighbors large enough to hold all of them.
    # Use 1024 to cover both geometries with margin.
    max_neighbors = 1024
    model = load_model_with_max_neighbors(ckpt, max_neighbors, device)

    for geometry in geometries:
        print(f"\n{'='*70}")
        print(f"  Geometry: {geometry}  (constant density = {TRAINING_DENSITY[geometry]:.5f})")
        print(f"{'='*70}")
        print(f"{'N':>8s} | {'L/spread':>10s} | {'rel_error':>10s} | {'cosine_sim':>10s} | {'MSE':>12s} | {'MAE':>10s}")
        print("-" * 70)

        for N in particle_counts:
            torch.cuda.empty_cache()
            box_size, spread, exp_neighbors = compute_box_for_density(geometry, N, cutoff)
            size_str = f"{box_size:.1f}" if geometry == "pse_periodic" else f"{spread:.1f}"

            try:
                data, target = make_sample(
                    geometry, N, device,
                    viscosity=data_cfg.viscosity,
                    a=data_cfg.hydrodynamic_radius,
                    box_size=box_size,
                    spread=spread,
                    seed=999 + N,
                )
                with torch.no_grad():
                    pred = model(data)
                metrics = compute_metrics(pred, target)
                print(
                    f"{N:>8d} | {size_str:>10s} | {metrics['rel_error']:>10.4f} | "
                    f"{metrics['cosine_sim']:>10.4f} | {metrics['mse']:>12.6f} | "
                    f"{metrics['mae']:>10.6f}"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"{N:>8d} | {size_str:>10s} | OOM")
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"{N:>8d} | {size_str:>10s} | Error: {e}")
                break


if __name__ == "__main__":
    main()
