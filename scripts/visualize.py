#!/usr/bin/env python3
"""Visualization script for hydrodynamics ML results.

Usage:
    python scripts/visualize.py --run-name torchmd_gn_single_geometry
    python scripts/visualize.py --run-name torchmd_gn_single_geometry --plot training_curves
    python scripts/visualize.py --compare torchmd_gn_single_geometry torchmd_et_single_geometry
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_CHOICES = [
    "training_curves",
    "per_geometry_val_losses",
    "generalization_matrix",
    "scaling",
    "rollout",
    "error_distributions",
    "model_comparison",
    "tb_training_curves",
    "pair_sweep_both_geometries",
    "nbody_accuracy_vs_N",
    "pse_accuracy_vs_boxsize",
]


def _setup_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def _save_fig(fig, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png")
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved {name}.png / .pdf")


def _read_history(results_dir: Path):
    """Read history.csv, returning list of dicts with float values (empty -> None)."""
    path = results_dir / "history.csv"
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v) if v.strip() != "" else None
                except ValueError:
                    parsed[k] = None
            rows.append(parsed)
    return rows


def plot_training_curves(results_dir: Path, output_dir: Path):
    rows = _read_history(results_dir)
    if rows is None:
        print("  Skipping training_curves: no history.csv")
        return

    epochs = [r["epoch"] for r in rows]
    train_loss = [r.get("train_loss") for r in rows]
    val_avg = [(r.get("val_avg")) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=1.5)

    val_epochs = [e for e, v in zip(epochs, val_avg) if v is not None]
    val_vals = [v for v in val_avg if v is not None]
    if val_vals:
        ax.plot(val_epochs, val_vals, label="Val Loss (avg)", linewidth=1.5, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Training Curves")
    _save_fig(fig, output_dir, "training_curves")


def plot_per_geometry_val_losses(results_dir: Path, output_dir: Path):
    rows = _read_history(results_dir)
    if rows is None:
        print("  Skipping per_geometry_val_losses: no history.csv")
        return

    val_cols = [k for k in rows[0] if k.startswith("val_") and k != "val_avg"]
    if not val_cols:
        print("  Skipping per_geometry_val_losses: no per-geometry val columns")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in val_cols:
        geo_name = col.replace("val_", "")
        vals = [(r["epoch"], r[col]) for r in rows if r.get(col) is not None]
        if vals:
            e, v = zip(*vals)
            ax.plot(e, v, label=geo_name, linewidth=1.5, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Per-Geometry Validation Loss")
    _save_fig(fig, output_dir, "per_geometry_val_losses")


def plot_generalization_matrix(results_dirs: list, output_dir: Path):
    """Heatmap of MSE: rows = training run, cols = test geometry."""
    matrix_data = {}
    all_geos = set()

    for rd in results_dirs:
        path = rd / "eval_single_step.json"
        if not path.exists():
            path = rd / "eval_generalization.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        run_name = rd.name
        matrix_data[run_name] = {}
        for geo, metrics in data.items():
            if geo == "generalization_summary":
                continue
            if isinstance(metrics, dict) and "mse" in metrics:
                matrix_data[run_name][geo] = metrics["mse"]
                all_geos.add(geo)

    if not matrix_data or not all_geos:
        print("  Skipping generalization_matrix: no data")
        return

    geos = sorted(all_geos)
    runs = sorted(matrix_data.keys())
    matrix = np.full((len(runs), len(geos)), np.nan)
    for i, run in enumerate(runs):
        for j, geo in enumerate(geos):
            matrix[i, j] = matrix_data[run].get(geo, np.nan)

    fig, ax = plt.subplots(figsize=(max(6, len(geos) * 2), max(4, len(runs) * 1.2)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(geos)))
    ax.set_xticklabels(geos, rotation=45, ha="right")
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(runs)

    for i in range(len(runs)):
        for j in range(len(geos)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2e}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="MSE")
    ax.set_title("Generalization Matrix")
    _save_fig(fig, output_dir, "generalization_matrix")


def plot_scaling(results_dir: Path, output_dir: Path):
    path = results_dir / "eval_scaling.json"
    if not path.exists():
        print("  Skipping scaling: no eval_scaling.json")
        return

    with open(path) as f:
        data = json.load(f)

    ns = sorted(int(k) for k in data.keys())
    times = [data[str(n)]["time_ms"] for n in ns]
    mems = [data[str(n)]["memory_mb"] for n in ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Time
    ax1.loglog(ns, times, "o-", linewidth=1.5)
    if len(ns) >= 2:
        coeffs = np.polyfit(np.log(ns), np.log(times), 1)
        fit_times = np.exp(np.polyval(coeffs, np.log(ns)))
        ax1.loglog(ns, fit_times, "--", alpha=0.7, label=f"slope={coeffs[0]:.2f}")
        ax1.legend()
    ax1.set_xlabel("Number of Particles (N)")
    ax1.set_ylabel("Inference Time (ms)")
    ax1.set_title("Scaling: Time")

    # Memory
    ax2.loglog(ns, mems, "s-", linewidth=1.5, color="tab:orange")
    if len(ns) >= 2:
        coeffs = np.polyfit(np.log(ns), np.log(mems), 1)
        fit_mems = np.exp(np.polyval(coeffs, np.log(ns)))
        ax2.loglog(ns, fit_mems, "--", alpha=0.7, color="tab:orange", label=f"slope={coeffs[0]:.2f}")
        ax2.legend()
    ax2.set_xlabel("Number of Particles (N)")
    ax2.set_ylabel("Memory (MB)")
    ax2.set_title("Scaling: Memory")

    fig.tight_layout()
    _save_fig(fig, output_dir, "scaling")


def plot_rollout(results_dir: Path, output_dir: Path):
    path = results_dir / "eval_rollout.json"
    if not path.exists():
        print("  Skipping rollout: no eval_rollout.json")
        return

    with open(path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 5), squeeze=False)

    for idx, (geo, metrics) in enumerate(data.items()):
        ax = axes[0, idx]
        steps = range(1, len(metrics["per_step_error"]) + 1)
        ax.plot(steps, metrics["per_step_error"], label="Per-step RMSE", linewidth=1.5)
        ax.plot(steps, metrics["cumulative_error"], label="Cumulative RMSE", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("RMSE")
        ax.set_title(f"Rollout: {geo}")
        ax.legend()

    fig.tight_layout()
    _save_fig(fig, output_dir, "rollout")

    # 3D position scatter if rollout positions file exists
    traj_path = results_dir / "eval_rollout_positions.npz"
    if traj_path.exists():
        traj_data = np.load(traj_path)
        geos = set(k.replace("_pred", "").replace("_true", "") for k in traj_data.files)
        for geo in sorted(geos):
            pred_key = f"{geo}_pred"
            true_key = f"{geo}_true"
            if pred_key not in traj_data or true_key not in traj_data:
                continue
            pred = traj_data[pred_key]
            true = traj_data[true_key]

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            # Plot final timestep
            ax.scatter(*pred[-1].T, s=10, alpha=0.6, label="Predicted")
            ax.scatter(*true[-1].T, s=10, alpha=0.6, label="True")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Rollout Trajectory (final step): {geo}")
            ax.legend()
            _save_fig(fig, output_dir, f"rollout_3d_{geo}")


def plot_error_distributions(results_dir: Path, output_dir: Path):
    path = results_dir / "eval_single_step.json"
    if not path.exists():
        print("  Skipping error_distributions: no eval_single_step.json")
        return

    with open(path) as f:
        data = json.load(f)

    geos = sorted(data.keys())
    metrics = ["mse", "mae", "relative_error"]
    x = np.arange(len(geos))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(geos) * 2.5), 5))
    for i, metric in enumerate(metrics):
        vals = [data[geo].get(metric, 0) for geo in geos]
        ax.bar(x + i * width, vals, width, label=metric.upper().replace("_", " "))

    ax.set_xticks(x + width)
    ax.set_xticklabels(geos, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Error Metrics by Geometry")
    fig.tight_layout()
    _save_fig(fig, output_dir, "error_distributions")


def plot_model_comparison(results_dirs: list, output_dir: Path):
    run_names = []
    best_losses = []

    for rd in results_dirs:
        path = rd / "summary.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        run_names.append(data.get("run_name", rd.name))
        best_losses.append(data.get("best_val_loss", float("nan")))

    if not run_names:
        print("  Skipping model_comparison: no summary.json files")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(run_names) * 1.5), 5))
    ax.bar(range(len(run_names)), best_losses, color="steelblue")
    ax.set_xticks(range(len(run_names)))
    ax.set_xticklabels(run_names, rotation=45, ha="right")
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Model Comparison")
    fig.tight_layout()
    _save_fig(fig, output_dir, "model_comparison")


# ── Live-evaluation plots (load model + solver, generate data) ───────

def _find_latest_checkpoint(model_type="torchmd_tn"):
    parent = Path("saved_models") / model_type
    runs = sorted(parent.glob("run_*/best.ckpt"))
    if not runs:
        return None
    return str(runs[-1])


def _load_model_and_config(device):
    """Load the latest TN checkpoint and return (model, data_cfg)."""
    import torch
    from models.lightning_wrapper import HydroLitModule
    from utils.config import DataConfig

    ckpt = _find_latest_checkpoint()
    if ckpt is None:
        print("  No checkpoint found for torchmd_tn")
        return None, None
    print(f"  Loading checkpoint: {ckpt}")
    lit = HydroLitModule.load_from_checkpoint(ckpt)
    lit.eval()
    lit.to(device)
    return lit.model, DataConfig()


def _make_model_fn(model, device, geometry):
    """Callable (positions, forces) -> displacement ndarray."""
    import torch
    from torch_geometric.data import Data
    from data.generate import GEOMETRY_IDS

    geo_id = GEOMETRY_IDS[geometry]

    def fn(positions, forces):
        N = positions.shape[0]
        data = Data(
            x=torch.ones(N, dtype=torch.long, device=device),
            pos=torch.tensor(positions, dtype=torch.float32, device=device),
            forces=torch.tensor(forces, dtype=torch.float32, device=device),
            geometry_id=torch.tensor(float(geo_id), device=device),
        )
        data.batch = torch.zeros(N, dtype=torch.long, device=device)
        return model(data).cpu().numpy()

    return fn


def _evaluate_cloud(model, device, geometry, N, data_cfg, num_samples=20,
                    seed=777, box_size_override=None):
    """Return (mean_rel_error, std_rel_error) over random cloud samples."""
    import torch
    from torch_geometric.data import Data
    from data.generate import (
        _sample_positions, _sample_forces, create_solver, GEOMETRY_IDS,
    )

    rng = np.random.default_rng(seed)
    geo_id = GEOMETRY_IDS[geometry]
    box_size = box_size_override if box_size_override is not None else data_cfg.box_size

    solver = create_solver(geometry, data_cfg.viscosity,
                           data_cfg.hydrodynamic_radius, box_size)
    rel_errors = []

    for _ in range(num_samples):
        positions = _sample_positions(N, geometry, rng,
                                      a=data_cfg.hydrodynamic_radius,
                                      box_size=box_size)
        forces = _sample_forces(N, data_cfg.force_scale, rng)
        solver.setPositions(positions)
        displacement, _ = solver.Mdot(forces=forces)
        displacement = np.array(displacement, dtype=np.float32).reshape(N, 3)

        L = box_size if geometry == "pse_periodic" else 1e6
        data = Data(
            x=torch.ones(N, dtype=torch.long, device=device),
            pos=torch.tensor(positions, dtype=torch.float32, device=device),
            forces=torch.tensor(forces, dtype=torch.float32, device=device),
            geometry_id=torch.tensor(float(geo_id), device=device),
            box_vecs=torch.tensor([[L, 0, 0], [0, L, 0], [0, 0, L]],
                                  dtype=torch.float32, device=device),
        )
        data.batch = torch.zeros(N, dtype=torch.long, device=device)
        data.num_nodes = N

        with torch.no_grad():
            pred = model(data).cpu().numpy()

        tnorm = np.maximum(np.linalg.norm(displacement, axis=-1, keepdims=True), 1e-8)
        rel = np.linalg.norm(pred - displacement, axis=-1, keepdims=True) / tnorm
        rel_errors.append(rel.mean())

    solver.clean()
    return float(np.mean(rel_errors)), float(np.std(rel_errors))


def plot_tb_training_curves(output_dir: Path, model_type="torchmd_tn"):
    """Read TensorBoard event files and plot train/val loss + per-geometry error."""
    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logs = sorted(glob.glob(f"lightning_logs/{model_type}/run_*/version_0"))
    if not logs:
        print("  Skipping tb_training_curves: no TensorBoard logs found")
        return
    ea = EventAccumulator(logs[-1])
    ea.Reload()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: train + val loss
    ax = axes[0]
    for tag, label, color in [("train_loss", "Train", "tab:blue"),
                               ("val_loss", "Validation", "tab:orange")]:
        if tag not in ea.Tags()["scalars"]:
            continue
        events = ea.Scalars(tag)
        vals = [e.value for e in events]
        if tag == "train_loss":
            n_val = len(ea.Scalars("val_loss")) if "val_loss" in ea.Tags()["scalars"] else 1
            spe = len(vals) / max(n_val, 1)
            xs = [s / spe for s in range(len(vals))]
            ax.plot(xs, vals, color=color, alpha=0.15, linewidth=0.5)
            w = max(1, len(vals) // 80)
            sm = np.convolve(vals, np.ones(w) / w, mode="valid")
            ax.plot(np.linspace(xs[0], xs[-1], len(sm)), sm,
                    color=color, linewidth=2, label=f"{label} (smoothed)")
        else:
            ax.plot(range(len(vals)), vals, color=color, linewidth=2, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (relative MSE)")
    ax.set_title("Training and Validation Loss")
    ax.set_yscale("log")
    ax.legend()

    # Right: per-geometry val relative error
    ax = axes[1]
    for tag, label, color in [("val_rel_error", "Overall", "black"),
                               ("val_rel_error_nbody", "NBody open", "tab:blue"),
                               ("val_rel_error_pse", "PSE periodic", "tab:orange")]:
        if tag not in ea.Tags()["scalars"]:
            continue
        vals = [e.value for e in ea.Scalars(tag)]
        ax.plot(range(len(vals)), vals, color=color, linewidth=2, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative Error")
    ax.set_title("Validation Relative Error by Geometry")
    ax.legend()

    fig.tight_layout()
    _save_fig(fig, output_dir, "tb_training_curves")


def plot_pair_sweep_both_geometries(output_dir: Path):
    """Pair sweep comparison for both NBody and PSE geometries."""
    import torch
    from evaluation.pair_sweep import evaluate_pair_sweep
    from evaluation.pair_sweep import plot_pair_sweep_both_particles, plot_pair_sweep_error
    from utils.solver import SolverCallable

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_cfg = _load_model_and_config(device)
    if model is None:
        return

    for geometry in ["nbody_open", "pse_periodic"]:
        print(f"  Pair sweep: {geometry} ...")
        model_fn = _make_model_fn(model, device, geometry)
        solver = SolverCallable(
            geometry=geometry, viscosity=data_cfg.viscosity,
            a=data_cfg.hydrodynamic_radius, box_size=data_cfg.box_size,
        )
        results = evaluate_pair_sweep(
            model_fn, solver, viscosity=data_cfg.viscosity,
            a=data_cfg.hydrodynamic_radius, box_size=data_cfg.box_size,
            periodic=(geometry == "pse_periodic"),
        )
        solver.clean()

        fig = plot_pair_sweep_both_particles(results)
        fig.suptitle(f"Pair Sweep: {geometry}", fontsize=14)
        fig.tight_layout()
        _save_fig(fig, output_dir, f"pair_sweep_{geometry}")

        fig = plot_pair_sweep_error(results)
        fig.suptitle(f"Pair Sweep Error: {geometry}", fontsize=14)
        fig.tight_layout()
        _save_fig(fig, output_dir, f"pair_sweep_error_{geometry}")


def plot_nbody_accuracy_vs_N(output_dir: Path):
    """NBody open: relative error vs number of particles N."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_cfg = _load_model_and_config(device)
    if model is None:
        return

    particle_counts = [8, 16, 32, 64, 128, 256, 512]
    means, stds, actual_counts = [], [], []

    for N in particle_counts:
        print(f"  NBody N={N} ...")
        try:
            m, s = _evaluate_cloud(model, device, "nbody_open", N, data_cfg,
                                    num_samples=20, seed=42 + N)
            means.append(m)
            stds.append(s)
            actual_counts.append(N)
            print(f"    rel_error = {m:.4f} +/- {s:.4f}")
        except Exception as e:
            print(f"    N={N} failed: {e}")
            break

    if not means:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(actual_counts, means, yerr=stds, fmt="o-", capsize=5,
                linewidth=2, markersize=8, color="tab:blue")
    ax.axvspan(0, 64, alpha=0.1, color="green", label="Training range (N <= 64)")
    ax.axvline(64, color="green", ls="--", alpha=0.5)
    ax.set_xlabel("Number of particles (N)")
    ax.set_ylabel("Mean relative error")
    ax.set_title("NBody Open: Accuracy vs Particle Count")
    ax.set_xscale("log", base=2)
    ax.set_xticks(actual_counts)
    ax.set_xticklabels([str(n) for n in actual_counts])
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, output_dir, "nbody_accuracy_vs_N")


def plot_pse_accuracy_vs_boxsize(output_dir: Path):
    """PSE periodic: relative error vs box size for N=16, 32, 64."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_cfg = _load_model_and_config(device)
    if model is None:
        return

    box_sizes = [16, 24, 32, 48, 64, 96, 128]
    particle_counts = [16, 32, 64]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for N, color in zip(particle_counts, colors):
        means, stds = [], []
        for box_size in box_sizes:
            print(f"  PSE N={N}, L={box_size} ...")
            m, s = _evaluate_cloud(model, device, "pse_periodic", N, data_cfg,
                                    num_samples=20, seed=123 + N + int(box_size),
                                    box_size_override=box_size)
            means.append(m)
            stds.append(s)
            print(f"    rel_error = {m:.4f} +/- {s:.4f}")

        ax.errorbar(box_sizes, means, yerr=stds, fmt="s-", capsize=4,
                    linewidth=2, markersize=7, color=color, label=f"N={N}")

    ax.axvline(32.0, color="green", ls="--", alpha=0.7, label="Training box size (L=32)")
    ax.set_xlabel("Box size (L)")
    ax.set_ylabel("Mean relative error")
    ax.set_title("PSE Periodic: Accuracy vs Box Size")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, output_dir, "pse_accuracy_vs_boxsize")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from training results")
    parser.add_argument("--run-name", type=str, help="Single run to plot")
    parser.add_argument("--compare", nargs="+", help="Multiple run names for comparison")
    parser.add_argument("--plot", type=str, default=None, choices=PLOT_CHOICES,
                        help="Generate only a specific plot")
    parser.add_argument("--results-root", type=str, default="results",
                        help="Root results directory (default: results)")
    args = parser.parse_args()

    if not args.run_name and not args.compare:
        parser.error("Provide --run-name or --compare")

    _setup_style()
    root = Path(args.results_root)

    if args.run_name:
        results_dir = root / args.run_name
        if not results_dir.exists():
            print(f"Error: {results_dir} not found")
            sys.exit(1)

        output_dir = results_dir / "plots"
        print(f"Generating plots for {args.run_name}...")

        single_run_plots = {
            "training_curves": plot_training_curves,
            "per_geometry_val_losses": plot_per_geometry_val_losses,
            "scaling": plot_scaling,
            "rollout": plot_rollout,
            "error_distributions": plot_error_distributions,
        }

        # Plots that only need output_dir (they load the model themselves)
        live_eval_plots = {
            "tb_training_curves": plot_tb_training_curves,
            "pair_sweep_both_geometries": plot_pair_sweep_both_geometries,
            "nbody_accuracy_vs_N": plot_nbody_accuracy_vs_N,
            "pse_accuracy_vs_boxsize": plot_pse_accuracy_vs_boxsize,
        }

        if args.plot:
            if args.plot in single_run_plots:
                single_run_plots[args.plot](results_dir, output_dir)
            elif args.plot in live_eval_plots:
                live_eval_plots[args.plot](output_dir)
            elif args.plot in ("generalization_matrix", "model_comparison"):
                print(f"  {args.plot} requires --compare mode")
            else:
                print(f"  Unknown plot: {args.plot}")
        else:
            for name, fn in single_run_plots.items():
                fn(results_dir, output_dir)
            for name, fn in live_eval_plots.items():
                fn(output_dir)

    if args.compare:
        results_dirs = [root / name for name in args.compare]
        missing = [rd for rd in results_dirs if not rd.exists()]
        if missing:
            print(f"Warning: missing directories: {[str(m) for m in missing]}")
            results_dirs = [rd for rd in results_dirs if rd.exists()]

        output_dir = root / "comparison_plots"
        print(f"Generating comparison plots for {args.compare}...")

        if args.plot is None or args.plot == "generalization_matrix":
            plot_generalization_matrix(results_dirs, output_dir)
        if args.plot is None or args.plot == "model_comparison":
            plot_model_comparison(results_dirs, output_dir)


if __name__ == "__main__":
    main()
