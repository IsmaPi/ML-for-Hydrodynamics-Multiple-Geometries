"""Diagnostic plot functions for hydrodynamic displacement models.

All plot functions take raw tensors and return a matplotlib Figure.
Called after training and during evaluation to save diagnostic PNGs.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


def _setup_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


# ── Data collection ──────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """Run model on dataloader and collect per-particle arrays."""
    model.eval()
    preds, targets, forces_list, pos_list = [], [], [], []
    geo_ids, num_nodes_list = [], []

    for batch in dataloader:
        batch = batch.to(device)
        pred = model(batch)
        preds.append(pred.cpu())
        targets.append(batch.y.cpu())
        forces_list.append(batch.forces.cpu())
        pos_list.append(batch.pos.cpu())
        geo_ids.append(batch.geometry_id[batch.batch].cpu())
        nn_per_graph = batch.ptr[1:] - batch.ptr[:-1]
        num_nodes_list.append(nn_per_graph[batch.batch].cpu())

    return {
        "pred": torch.cat(preds),
        "true": torch.cat(targets),
        "forces": torch.cat(forces_list),
        "pos": torch.cat(pos_list),
        "geometry_id": torch.cat(geo_ids),
        "num_nodes": torch.cat(num_nodes_list),
    }


def compute_derived_metrics(data):
    """Add derived metric fields to the data dict."""
    pred, true, forces = data["pred"].detach(), data["true"].detach(), data["forces"].detach()
    data["error_vec"] = pred - true
    data["error_norm"] = data["error_vec"].norm(dim=-1)
    data["true_norm"] = true.norm(dim=-1)
    data["pred_norm"] = pred.norm(dim=-1)
    data["force_norm"] = forces.norm(dim=-1)
    data["relative_error"] = data["error_norm"] / data["true_norm"].clamp(min=1e-8)
    data["cosine_sim"] = F.cosine_similarity(pred, true, dim=-1)
    data["is_active"] = data["force_norm"] > 0.1
    return data


# ── Plot functions ───────────────────────────────────────────────────

def plot_pred_vs_true_magnitude(data):
    """Hexbin scatter of predicted vs true displacement magnitude."""
    fig, ax = plt.subplots(figsize=(7, 6))

    true_mag = data["true_norm"].numpy()
    pred_mag = data["pred_norm"].numpy()
    geo = data["geometry_id"].numpy()

    nbody = geo == 0
    pse = geo == 1

    if nbody.any():
        ax.scatter(true_mag[nbody], pred_mag[nbody], s=1, alpha=0.03,
                   c="tab:blue", label="nbody_open", rasterized=True)
    if pse.any():
        ax.scatter(true_mag[pse], pred_mag[pse], s=1, alpha=0.03,
                   c="tab:orange", label="pse_periodic", rasterized=True)

    lims = [max(1e-6, min(true_mag.min(), pred_mag.min())),
            max(true_mag.max(), pred_mag.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.7, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True displacement magnitude")
    ax.set_ylabel("Predicted displacement magnitude")
    ax.set_title("Predicted vs True Magnitude")
    ax.legend(markerscale=10)
    fig.tight_layout()
    return fig


def plot_error_cdf(data):
    """CDF of relative error: all, active, passive."""
    fig, ax = plt.subplots(figsize=(7, 5))

    rel_err = data["relative_error"].numpy()
    active = data["is_active"].numpy()

    for subset, label, color, ls in [
        (np.ones_like(rel_err, dtype=bool), "All particles", "black", "-"),
        (active, "Active (|F|>0.1)", "tab:blue", "-"),
        (~active, "Passive (|F|<=0.1)", "tab:red", "--"),
    ]:
        vals = np.sort(rel_err[subset])
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, label=label, color=color, ls=ls, lw=1.5)

    for pct in [0.5, 0.9, 0.95]:
        ax.axhline(pct, color="gray", ls=":", alpha=0.3)
        ax.text(0.02, pct + 0.01, f"{int(pct*100)}th", fontsize=8, color="gray",
                transform=ax.get_yaxis_transform())

    ax.set_xlim(0, min(2.0, np.percentile(rel_err, 99.5)))
    ax.set_xlabel("Relative error")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Error CDF")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cosine_sim(data):
    """Histogram of cosine similarity (directional accuracy)."""
    cos_sim = data["cosine_sim"].numpy()
    true_norm = data["true_norm"].numpy()
    active = data["is_active"].numpy()

    valid = true_norm > 0.01

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(cos_sim[valid], bins=100, range=(-1, 1), density=True,
            alpha=0.7, color="steelblue", edgecolor="none")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Directional accuracy (all, |disp|>0.01)")
    ax.axvline(1.0, color="green", ls="--", alpha=0.5, label="Perfect")
    ax.axvline(0.0, color="red", ls=":", alpha=0.5, label="Orthogonal")
    ax.legend()

    ax = axes[1]
    active_valid = valid & active
    passive_valid = valid & ~active
    if active_valid.any():
        ax.hist(cos_sim[active_valid], bins=100, range=(-1, 1), density=True,
                alpha=0.6, label="Active", color="tab:blue", edgecolor="none")
    if passive_valid.any():
        ax.hist(cos_sim[passive_valid], bins=100, range=(-1, 1), density=True,
                alpha=0.6, label="Passive", color="tab:red", edgecolor="none")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Active vs Passive directional accuracy")
    ax.legend()

    fig.tight_layout()
    return fig


def plot_error_breakdown(data):
    """Violin plots of relative error by geometry x active/passive and by N."""
    rel_err = data["relative_error"].numpy()
    geo = data["geometry_id"].numpy()
    active = data["is_active"].numpy()
    nn = data["num_nodes"].numpy()

    clip_val = np.percentile(rel_err, 99)
    rel_clipped = np.clip(rel_err, 0, clip_val)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    groups = []
    labels = []
    for gid, gname in [(0, "nbody"), (1, "pse")]:
        for is_act, aname in [(True, "active"), (False, "passive")]:
            mask = (geo == gid) & (active == is_act)
            if mask.any():
                groups.append(rel_clipped[mask])
                labels.append(f"{gname}\n{aname}")

    parts = ax.violinplot(groups, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative error (clipped at 99th pct)")
    ax.set_title("Error by geometry and particle role")

    ax = axes[1]
    groups_n = []
    labels_n = []
    for n in [16, 32, 64]:
        mask = nn == n
        if mask.any():
            groups_n.append(rel_clipped[mask])
            labels_n.append(f"N={n}")

    if groups_n:
        parts = ax.violinplot(groups_n, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels_n) + 1))
        ax.set_xticklabels(labels_n)
    ax.set_ylabel("Relative error (clipped at 99th pct)")
    ax.set_title("Error by particle count")

    fig.tight_layout()
    return fig


def plot_pred_vs_true_components(data):
    """Scatter of predicted vs true per component (x, y, z)."""
    pred = data["pred"].numpy()
    true = data["true"].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ["x", "y", "z"]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        t = true[:, i]
        p = pred[:, i]
        ax.hexbin(t, p, gridsize=80, mincnt=1, cmap="viridis")
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_yscale("symlog", linthresh=1e-4)

        lims = [min(t.min(), p.min()) * 1.1, max(t.max(), p.max()) * 1.1]
        ax.plot(lims, lims, "r--", lw=1, alpha=0.7)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Predicted {label}")

        ss_res = np.sum((p - t) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.set_title(f"{label}-component (R²={r2:.4f})")

    fig.suptitle("Predicted vs True: Per Component", fontsize=13)
    fig.tight_layout()
    return fig


def plot_error_vs_force(data):
    """Hexbin of force magnitude vs relative error."""
    force_mag = data["force_norm"].numpy()
    rel_err = data["relative_error"].numpy()

    # Exclude zero-error points for log scale
    valid = (force_mag > 1e-8) & (rel_err > 1e-12)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hexbin(force_mag[valid], rel_err[valid], gridsize=80, mincnt=1,
              cmap="viridis", xscale="log", yscale="log")
    ax.axvline(0.1, color="red", ls="--", alpha=0.7, label="|F|=0.1 threshold")
    ax.set_xlabel("Force magnitude")
    ax.set_ylabel("Relative error")
    ax.set_title("Error vs Force Magnitude")
    ax.legend()
    fig.colorbar(ax.collections[0], ax=ax, label="Count")
    fig.tight_layout()
    return fig


def plot_spatial_error_map(dataset, device, model, num_per_combo=1):
    """Particle scatter colored by error with displacement arrows.

    Shows representative samples: 1 per geometry x 3 particle counts.
    Returns dict of {figure_name: fig}.
    """
    target_combos = [(0, 16), (0, 32), (0, 64), (1, 16), (1, 32), (1, 64)]
    figs = {}

    for geo_target, n_target in target_combos:
        found = False
        for idx in range(min(len(dataset), 5000)):
            sample = dataset[idx]
            n = sample.num_nodes
            geo = sample.geometry_id.item()
            if n == n_target and int(geo) == geo_target:
                found = True
                break

        if not found:
            continue

        batch = Batch.from_data_list([sample]).to(device)
        with torch.no_grad():
            pred = model(batch).cpu().numpy()
        true_disp = sample.y.numpy()
        pos = sample.pos.numpy()

        err = np.linalg.norm(pred - true_disp, axis=-1)
        true_mag = np.linalg.norm(true_disp, axis=-1)
        rel_err = err / np.maximum(true_mag, 1e-8)

        geo_name = "nbody_open" if geo_target == 0 else "pse_periodic"
        fig_name = f"spatial_error_map_{geo_name}_N{n_target}"

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        scale = 3.0
        vmax = min(1.0, np.percentile(rel_err, 95))

        # XY projection
        ax = axes[0]
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=rel_err, cmap="RdYlGn_r",
                        s=80, edgecolors="black", linewidths=0.5, vmin=0, vmax=vmax)
        ax.quiver(pos[:, 0], pos[:, 1], true_disp[:, 0], true_disp[:, 1],
                  color="blue", alpha=0.6, scale=scale, width=0.004,
                  label="True displacement")
        ax.quiver(pos[:, 0], pos[:, 1], pred[:, 0], pred[:, 1],
                  color="red", alpha=0.6, scale=scale, width=0.004,
                  label="Predicted displacement")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("XY projection")
        ax.legend(fontsize=8)
        fig.colorbar(sc, ax=ax, label="Relative error")
        ax.set_aspect("equal")

        # XZ projection
        ax = axes[1]
        sc = ax.scatter(pos[:, 0], pos[:, 2], c=rel_err, cmap="RdYlGn_r",
                        s=80, edgecolors="black", linewidths=0.5, vmin=0, vmax=vmax)
        ax.quiver(pos[:, 0], pos[:, 2], true_disp[:, 0], true_disp[:, 2],
                  color="blue", alpha=0.6, scale=scale, width=0.004,
                  label="True displacement")
        ax.quiver(pos[:, 0], pos[:, 2], pred[:, 0], pred[:, 2],
                  color="red", alpha=0.6, scale=scale, width=0.004,
                  label="Predicted displacement")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title("XZ projection")
        ax.legend(fontsize=8)
        fig.colorbar(sc, ax=ax, label="Relative error")
        ax.set_aspect("equal")

        mean_rel = np.mean(rel_err)
        fig.suptitle(f"{geo_name} N={n_target} — mean rel error: {mean_rel:.1%}",
                     fontsize=13)
        fig.tight_layout()
        figs[fig_name] = fig

    return figs


def plot_absolute_error_hist(data):
    """Histogram of absolute error split by active/passive and geometry."""
    abs_err = data["error_norm"].numpy()
    active = data["is_active"].numpy()
    geo = data["geometry_id"].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    clip = np.percentile(abs_err, 99)

    ax = axes[0]
    if active.any():
        ax.hist(abs_err[active], bins=80, range=(0, clip), density=True,
                alpha=0.6, label="Active", color="tab:blue", edgecolor="none")
    if (~active).any():
        ax.hist(abs_err[~active], bins=80, range=(0, clip), density=True,
                alpha=0.6, label="Passive", color="tab:red", edgecolor="none")
    ax.set_xlabel("Absolute error (L2 norm)")
    ax.set_ylabel("Density")
    ax.set_title("Absolute error: Active vs Passive")
    ax.legend()

    ax = axes[1]
    for gid, gname, color in [(0, "nbody_open", "tab:blue"), (1, "pse_periodic", "tab:orange")]:
        mask = geo == gid
        if mask.any():
            ax.hist(abs_err[mask], bins=80, range=(0, clip), density=True,
                    alpha=0.6, label=gname, color=color, edgecolor="none")
    ax.set_xlabel("Absolute error (L2 norm)")
    ax.set_ylabel("Density")
    ax.set_title("Absolute error: by Geometry")
    ax.legend()

    fig.tight_layout()
    return fig


# ── Generate and save all plots ──────────────────────────────────────

def generate_all_plots(model, datamodule, device, results_dir, tb_writer=None,
                       global_step=0):
    """Generate all diagnostic plots, save PNGs, and log to TensorBoard.

    Called after training+testing completes in run.py.

    Args:
        model: trained model in eval mode
        datamodule: HydroDataModule with setup() already called
        device: torch device
        results_dir: Path to save PNGs (old PNGs cleaned first)
        tb_writer: TensorBoard SummaryWriter (from trainer.logger.experiment)
        global_step: step for TensorBoard logging
    """
    _setup_style()
    results_dir = Path(results_dir)

    # Clean old PNGs
    if results_dir.exists():
        for f in results_dir.glob("*.png"):
            f.unlink()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots for each split: train, val, test
    splits = {
        "train": datamodule.train_dataset,
        "val": datamodule.val_dataset,
        "test": datamodule.test_dataset,
    }

    total_figs = 0
    for split_name, dataset in splits.items():
        split_dir = results_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        data = collect_predictions(model, loader, device)
        data = compute_derived_metrics(data)

        plots = {
            "pred_vs_true_magnitude": plot_pred_vs_true_magnitude(data),
            "error_cdf": plot_error_cdf(data),
            "cosine_sim_hist": plot_cosine_sim(data),
            "error_breakdown": plot_error_breakdown(data),
            "pred_vs_true_components": plot_pred_vs_true_components(data),
            "error_vs_force": plot_error_vs_force(data),
            "absolute_error_hist": plot_absolute_error_hist(data),
        }

        for name, fig in plots.items():
            fig.savefig(split_dir / f"{name}.png")
            if tb_writer is not None:
                tb_writer.add_figure(f"{split_name}/{name}", fig,
                                     global_step=global_step)
            plt.close(fig)

        # Spatial error maps (multiple figures)
        spatial_figs = plot_spatial_error_map(dataset, device, model)
        for name, fig in spatial_figs.items():
            fig.savefig(split_dir / f"{name}.png")
            if tb_writer is not None:
                tb_writer.add_figure(f"{split_name}/{name}", fig,
                                     global_step=global_step)
            plt.close(fig)

        total_figs += len(plots) + len(spatial_figs)
        print(f"  Saved {len(plots) + len(spatial_figs)} plots to {split_dir}/")

    print(f"  Total: {total_figs} plots across train/val/test in {results_dir}/")
