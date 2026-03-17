"""Two-particle distance sweep evaluation.

Places two particles at varying distances, applies a force on particle 1,
and compares two callables (e.g. model vs libMobility ground truth).
"""

import logging
import math

import numpy as np
import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_pair_sweep(
    model_fn,
    solver_fn,
    d_min: float = 2.5,
    d_max: float = 15.0,
    num_points: int = 100,
    force_direction: np.ndarray = None,
    force_magnitude: float = 1.0,
    viscosity: float = 1.0,
    a: float = 1.0,
    box_size: float = 32.0,
    periodic: bool = False,
):
    """Run a two-particle distance sweep comparing two displacement functions.

    Args:
        model_fn: callable (positions_ndarray, forces_ndarray) -> displacement_ndarray
        solver_fn: callable (positions_ndarray, forces_ndarray) -> displacement_ndarray
        d_min: minimum distance (units of a)
        d_max: maximum distance (units of a)
        num_points: number of distance points
        force_direction: (3,) force direction on particle 1 (default: x-axis)
        force_magnitude: magnitude of force on particle 1
        viscosity: fluid viscosity (for self_mobility reference line)
        a: hydrodynamic radius (for self_mobility reference line)
        box_size: box size for periodic geometry (position placement)
        periodic: whether to center particles in box

    Returns:
        dict with:
            distances: (num_points,) array of distances
            pred_displacements: (num_points, 2, 3) model predictions
            true_displacements: (num_points, 2, 3) ground truth
            self_mobility: analytical self-mobility value
            force_magnitude: applied force magnitude
    """
    if force_direction is None:
        force_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    force_direction = force_direction / np.linalg.norm(force_direction)

    distances = np.linspace(d_min, d_max, num_points)
    pred_all = []
    true_all = []

    forces = np.array(
        [force_direction * force_magnitude, [0.0, 0.0, 0.0]], dtype=np.float32
    )

    for d in distances:
        if periodic:
            center = np.array([box_size / 2] * 3, dtype=np.float32)
            positions = np.array([center, center + np.array([d, 0, 0], dtype=np.float32)])
        else:
            positions = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=np.float32)

        true_all.append(solver_fn(positions, forces))
        pred_all.append(model_fn(positions, forces))

    self_mobility = 1.0 / (6.0 * math.pi * viscosity * a)

    return {
        "distances": distances,
        "pred_displacements": np.stack(pred_all),      # (num_points, 2, 3)
        "true_displacements": np.stack(true_all),       # (num_points, 2, 3)
        "self_mobility": self_mobility,
        "force_magnitude": force_magnitude,
    }


def plot_pair_sweep(results, save_path=None, particle_idx=0):
    """Plot predicted vs true displacement for a pair sweep.

    Args:
        results: dict from evaluate_pair_sweep
        save_path: optional path to save figure
        particle_idx: which particle to plot (0 or 1)

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    distances = results["distances"]
    pred = results["pred_displacements"][:, particle_idx, :]
    true = results["true_displacements"][:, particle_idx, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ["x", "y", "z"]

    self_mob = results.get("self_mobility")
    f_mag = results.get("force_magnitude", 1.0)

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(distances, pred[:, i], label=f"{label} (predicted)", linewidth=2)
        ax.plot(distances, true[:, i], "--", label=f"{label} (ground truth)", linewidth=2)
        if self_mob is not None and particle_idx == 0:
            baseline_val = self_mob * f_mag if label == "x" else 0.0
            ax.axhline(y=baseline_val, color="gray", linestyle=":", alpha=0.7,
                        label="self-mobility baseline")
        ax.set_xlabel("Distance (units of a)")
        ax.set_ylabel("Displacement")
        ax.set_title(f"Particle {particle_idx + 1}: {label}-component")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Pair Sweep: Model vs Ground Truth", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info("Pair sweep plot saved to %s", save_path)

    return fig


def plot_pair_sweep_both_particles(results, save_path=None):
    """Plot pair sweep for both particles.

    Args:
        results: dict from evaluate_pair_sweep
        save_path: optional path to save figure

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    distances = results["distances"]
    pred = results["pred_displacements"]
    true = results["true_displacements"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    labels = ["x", "y", "z"]

    for pidx, row in enumerate(axes):
        for i, (ax, label) in enumerate(zip(row, labels)):
            ax.plot(distances, pred[:, pidx, i], label="predicted", linewidth=2)
            ax.plot(distances, true[:, pidx, i], "--", label="ground truth", linewidth=2)
            ax.set_xlabel("Distance (units of a)")
            ax.set_ylabel("Displacement")
            ax.set_title(f"Particle {pidx + 1}: {label}-component")
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.suptitle("Pair Sweep: Both Particles", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info("Both-particles pair sweep saved to %s", save_path)

    return fig


def plot_pair_sweep_error(results, save_path=None):
    """Plot absolute and relative error vs distance.

    Args:
        results: dict from evaluate_pair_sweep
        save_path: optional path to save figure

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    distances = results["distances"]
    pred = results["pred_displacements"]
    true = results["true_displacements"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for pidx in range(2):
        abs_err = np.abs(pred[:, pidx, :] - true[:, pidx, :])
        true_norm = np.linalg.norm(true[:, pidx, :], axis=-1, keepdims=True)
        true_norm = np.maximum(true_norm, 1e-12)
        rel_err = np.linalg.norm(pred[:, pidx, :] - true[:, pidx, :], axis=-1, keepdims=True) / true_norm

        axes[0].plot(distances, abs_err.sum(axis=-1), label=f"Particle {pidx+1}", linewidth=2)
        axes[1].plot(distances, rel_err.squeeze() * 100, label=f"Particle {pidx+1}", linewidth=2)

    axes[0].set_xlabel("Distance (units of a)")
    axes[0].set_ylabel("Absolute Error (sum of components)")
    axes[0].set_title("Absolute Error vs Distance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Distance (units of a)")
    axes[1].set_ylabel("Relative Error (%)")
    axes[1].set_title("Relative Error vs Distance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info("Error plot saved to %s", save_path)

    return fig
