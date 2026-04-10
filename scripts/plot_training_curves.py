#!/usr/bin/env python3
"""Generate training curve plots from TensorBoard event logs.

Usage:
    python scripts/plot_training_curves.py                          # latest run
    python scripts/plot_training_curves.py --run run_030            # specific run
    python scripts/plot_training_curves.py --output figures/        # custom output dir
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _setup_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def read_tb_scalars(log_dir, tag):
    """Read a scalar tag from a TensorBoard event file."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def steps_to_epochs(steps, values, steps_per_epoch):
    """Convert global steps to epoch numbers."""
    return [s / steps_per_epoch for s in steps], values


def smooth(values, window):
    """Simple moving average."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_loss_curves(log_dir, output_dir):
    """Plot train/val loss and per-geometry val relative error."""
    _setup_style()

    # Read data
    train_steps, train_vals = read_tb_scalars(log_dir, "train_loss")
    val_steps, val_vals = read_tb_scalars(log_dir, "val_loss")

    if not train_vals or not val_vals:
        print("No training data found.")
        return

    # Estimate steps per epoch from validation frequency
    steps_per_epoch = val_steps[1] - val_steps[0] if len(val_steps) > 1 else len(train_vals)

    # Convert to epochs
    train_epochs = [s / steps_per_epoch for s in train_steps]
    val_epochs = [s / steps_per_epoch for s in val_steps]

    # ── Figure 1: Train + Val Loss ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    # Raw train loss (transparent)
    ax.plot(train_epochs, train_vals, color="tab:blue", alpha=0.15, linewidth=0.5)
    # Smoothed train loss
    w = max(1, len(train_vals) // 80)
    sm = smooth(train_vals, w)
    sm_epochs = np.linspace(train_epochs[0], train_epochs[-1], len(sm))
    ax.plot(sm_epochs, sm, color="tab:blue", linewidth=2, label="Train (smoothed)")
    # Val loss
    ax.plot(val_epochs, val_vals, color="tab:orange", linewidth=2, label="Validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (relative MSE)")
    ax.set_title("Training and Validation Loss")
    ax.set_yscale("log")
    ax.legend()

    # ── Figure 2: Per-geometry val relative error ────────────────────
    ax = axes[1]
    geo_tags = [
        ("val_rel_error", "Overall", "black"),
        ("val_rel_error_nbody", "NBody (open)", "tab:blue"),
        ("val_rel_error_pse", "PSE (periodic)", "tab:orange"),
    ]
    for tag, label, color in geo_tags:
        steps, vals = read_tb_scalars(log_dir, tag)
        if vals:
            epochs = [s / steps_per_epoch for s in steps]
            ax.plot(epochs, vals, color=color, linewidth=2, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative Error")
    ax.set_title("Validation Relative Error by Geometry")
    ax.legend()

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_curves.png"
    fig.savefig(path)
    fig.savefig(output_dir / "training_curves.pdf")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 3: Per-N val relative error ───────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    n_tags = [
        ("val_rel_error_N16", "N=16", "tab:blue"),
        ("val_rel_error_N32", "N=32", "tab:orange"),
        ("val_rel_error_N64", "N=64", "tab:green"),
    ]
    for tag, label, color in n_tags:
        steps, vals = read_tb_scalars(log_dir, tag)
        if vals:
            epochs = [s / steps_per_epoch for s in steps]
            ax2.plot(epochs, vals, color=color, linewidth=2, label=label)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Validation Relative Error by Particle Count")
    ax2.legend()
    fig2.tight_layout()
    path2 = output_dir / "training_curves_per_N.png"
    fig2.savefig(path2)
    fig2.savefig(output_dir / "training_curves_per_N.pdf")
    plt.close(fig2)
    print(f"Saved: {path2}")

    # ── Print summary ────────────────────────────────────────────────
    print(f"\nTraining summary:")
    print(f"  Epochs: {len(val_vals)}")
    print(f"  Best val_loss: {min(val_vals):.6f}")
    for tag, label, _ in geo_tags:
        _, vals = read_tb_scalars(log_dir, tag)
        if vals:
            print(f"  Best {label}: {min(vals):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from TensorBoard logs")
    parser.add_argument("--run", type=str, default=None,
                        help="Specific run name (e.g. run_030). Default: latest.")
    parser.add_argument("--model", type=str, default="torchmd_tn",
                        help="Model type (default: torchmd_tn)")
    parser.add_argument("--output", type=str, default="results/torchmd_tn/training_curves",
                        help="Output directory for plots")
    args = parser.parse_args()

    # Find log directory
    if args.run:
        log_dir = f"lightning_logs/{args.model}/{args.run}/version_0"
    else:
        logs = sorted(glob.glob(f"lightning_logs/{args.model}/run_*/version_0"))
        if not logs:
            print(f"No logs found for {args.model}")
            return
        log_dir = logs[-1]
        print(f"Using latest: {log_dir}")

    plot_loss_curves(log_dir, args.output)


if __name__ == "__main__":
    main()
