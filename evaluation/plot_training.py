"""Plot training curves from TensorBoard logs.

Reads event files and generates matplotlib figures for loss, metrics,
and learning rate over training.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


def _read_tb_scalars(log_dir: str):
    """Read scalar summaries from TensorBoard event files."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir)
    ea.Reload()

    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        scalars[tag] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
        }
    return scalars


def _find_latest_tb_dir(model_type: str) -> str:
    """Find the latest TensorBoard log directory for a model."""
    log_root = Path("logs") / model_type
    if not log_root.exists():
        raise FileNotFoundError(f"No logs found at {log_root}")

    # TensorBoard logger creates logs/<name>/version_N/
    versions = sorted(log_root.glob("run_*/version_*"))
    if not versions:
        versions = sorted(log_root.glob("version_*"))
    if not versions:
        raise FileNotFoundError(f"No version directories in {log_root}")

    return str(versions[-1])


def plot_training_curves(model_type: str, save_dir: str = None):
    """Plot training curves from TensorBoard logs.

    Generates:
    1. Loss curves (train + val + baseline)
    2. Validation metrics (rel_error, improvement, per-component MAE)
    3. Learning rate schedule
    4. Correction magnitude over training
    """
    tb_dir = _find_latest_tb_dir(model_type)
    scalars = _read_tb_scalars(tb_dir)

    if save_dir is None:
        save_dir = f"results/{model_type}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    if "train_loss" in scalars:
        s = scalars["train_loss"]
        ax.plot(s["steps"], s["values"], label="Train Loss", alpha=0.7)
    if "val_loss" in scalars:
        s = scalars["val_loss"]
        ax.plot(s["steps"], s["values"], label="Val Loss", linewidth=2)
    for tag in ["val_baseline_loss", "val_baseline_mse"]:
        if tag in scalars:
            s = scalars[tag]
            ax.plot(s["steps"], s["values"], "--", label="Baseline (self-mobility only)", linewidth=2)
            break
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_type}: Training & Validation Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Loss curves saved to %s/loss_curves.png", save_dir)

    # 2. Validation metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    if "val_rel_error" in scalars:
        s = scalars["val_rel_error"]
        axes[0, 0].plot(s["steps"], s["values"])
        axes[0, 0].set_title("Relative Error")
        axes[0, 0].set_ylabel("Relative Error")
        axes[0, 0].grid(True, alpha=0.3)

    if "val_improvement" in scalars:
        s = scalars["val_improvement"]
        axes[0, 1].plot(s["steps"], s["values"])
        axes[0, 1].axhline(y=0, color="r", linestyle="--", alpha=0.5, label="No improvement")
        axes[0, 1].set_title("Improvement over Baseline")
        axes[0, 1].set_ylabel("Fraction")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Per-component MAE
    for comp, color in [("val_mae_x", "C0"), ("val_mae_y", "C1"), ("val_mae_z", "C2")]:
        if comp in scalars:
            s = scalars[comp]
            axes[1, 0].plot(s["steps"], s["values"], label=comp.split("_")[-1], color=color)
    axes[1, 0].set_title("Per-Component MAE")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Correction RMS
    for tag, label in [("train_correction_rms", "Train"), ("val_correction_rms", "Val")]:
        if tag in scalars:
            s = scalars[tag]
            axes[1, 1].plot(s["steps"], s["values"], label=label)
    axes[1, 1].set_title("Correction Magnitude (RMS)")
    axes[1, 1].set_ylabel("RMS")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Step")

    fig.suptitle(f"{model_type}: Validation Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/val_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Validation metrics saved to %s/val_metrics.png", save_dir)

    # 3. Learning rate
    if "lr" in scalars:
        fig, ax = plt.subplots(figsize=(8, 4))
        s = scalars["lr"]
        ax.plot(s["steps"], s["values"])
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"{model_type}: Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{save_dir}/lr_schedule.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("LR schedule saved to %s/lr_schedule.png", save_dir)
