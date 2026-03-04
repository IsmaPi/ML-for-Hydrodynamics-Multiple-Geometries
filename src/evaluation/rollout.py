"""Autoregressive rollout evaluation.

Iterates Q^{n+1} = Q^n + DeltaX_hat^n using model predictions
and compares against ground-truth trajectory from libMobility (or synthetic).
"""

import torch
import numpy as np
from typing import Optional
from torch_geometric.data import Data

from ..data.geometry import build_geometry_params, get_geometry_constraints
from ..data.graph_construction import build_graph
from ..data.generate import (
    sample_forces, enforce_boundaries,
    generate_trajectory,
)
from ..data.normalization import FeatureNormalizer


@torch.no_grad()
def evaluate_rollout(
    model: torch.nn.Module,
    geometry_key: str,
    config: dict,
    normalizer: Optional[FeatureNormalizer],
    N: int,
    num_steps: int,
    dt: float,
    device: torch.device,
    graph_method: str = "knn",
    k_neighbors: int = 16,
    cutoff_radius: float = 5.0,
    use_synthetic: bool = True,
    seed: int = 123,
) -> dict:
    """Run autoregressive rollout and compare to ground truth.

    Returns:
        pred_trajectory: (num_steps+1, N, 3)
        true_trajectory: (num_steps+1, N, 3)
        per_step_error: (num_steps,) RMSE per step
        cumulative_error: (num_steps,) cumulative RMSE
    """
    model.eval()
    rng = np.random.default_rng(seed)
    force_scale = config.get("force_scale", 1.0)

    # Generate ground truth trajectory
    true_traj = generate_trajectory(
        geometry_key=geometry_key,
        config=config,
        N=N,
        T_steps=num_steps,
        dt=dt,
        force_scale=force_scale,
        rng=np.random.default_rng(seed),  # same seed for same initial conditions
        use_synthetic=use_synthetic,
    )

    theta = build_geometry_params(geometry_key, config).to(device)
    theta_expanded = theta.unsqueeze(0).expand(N, -1)

    # Model rollout using the SAME initial conditions and forces
    pred_positions = np.zeros((num_steps + 1, N, 3))
    pred_positions[0] = true_traj["positions"][0]
    prev_velocity = torch.zeros(N, 3, device=device)

    for n in range(num_steps):
        pos_np = pred_positions[n]
        forces_np = true_traj["forces"][n]  # same forces as ground truth

        pos_t = torch.tensor(pos_np, dtype=torch.float32, device=device)
        forces_t = torch.tensor(forces_np, dtype=torch.float32, device=device)

        # Build features
        if normalizer:
            pos_norm = normalizer.transform(pos_t.cpu(), "positions").to(device)
            forces_norm = normalizer.transform(forces_t.cpu(), "forces").to(device)
            prev_vel_norm = normalizer.transform(prev_velocity.cpu(), "velocities").to(device)
        else:
            pos_norm = pos_t
            forces_norm = forces_t
            prev_vel_norm = prev_velocity

        node_features = torch.cat([pos_norm, forces_norm, prev_vel_norm, theta_expanded], dim=-1)

        edge_index, edge_attr = build_graph(pos_t, graph_method, k_neighbors, cutoff_radius)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos_t,
        ).to(device)
        data.batch = torch.zeros(N, dtype=torch.long, device=device)

        pred_disp = model(data)

        if normalizer:
            pred_disp = normalizer.inverse_transform(pred_disp.cpu(), "displacements").numpy()
        else:
            pred_disp = pred_disp.cpu().numpy()

        pred_positions[n + 1] = pred_positions[n] + pred_disp
        pred_positions[n + 1] = enforce_boundaries(pred_positions[n + 1], geometry_key, config)

        prev_velocity = torch.tensor(pred_disp / dt, dtype=torch.float32, device=device)

    true_positions = true_traj["positions"]

    # Per-step RMSE
    per_step_error = np.sqrt(
        ((pred_positions[1:] - true_positions[1:]) ** 2).sum(axis=-1).mean(axis=-1)
    )
    cumulative_error = np.cumsum(per_step_error)

    return {
        "pred_trajectory": pred_positions,
        "true_trajectory": true_positions,
        "per_step_error": per_step_error,
        "cumulative_error": cumulative_error,
    }
