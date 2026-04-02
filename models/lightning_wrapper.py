"""PyTorch Lightning module wrapping the hydrodynamic displacement models.

Handles training/validation step, loss computation, optimizer/scheduler setup.
Works with HydroTorchMD_ET, HydroTorchMD_TN, and HydroTorchMD_GN.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, SequentialLR, LinearLR

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from models.torchmd_et import HydroTorchMD_ET
from models.torchmd_gn import HydroTorchMD_GN
from models.torchmd_tn import HydroTorchMD_TN
from utils.config import ModelConfig, TrainingConfig

# Allow torch.load(weights_only=True) to deserialize our config dataclasses
torch.serialization.add_safe_globals([ModelConfig, TrainingConfig])


def build_model(model_cfg):
    """Instantiate model from config."""
    if model_cfg.model_type == "torchmd_et":
        return HydroTorchMD_ET(model_cfg)
    elif model_cfg.model_type == "torchmd_gn":
        return HydroTorchMD_GN(model_cfg)
    elif model_cfg.model_type == "torchmd_tn":
        return HydroTorchMD_TN(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_cfg.model_type}")


class HydroLitModule(pl.LightningModule):
    """Lightning module for hydrodynamic displacement prediction."""

    def __init__(self, model_cfg, train_cfg):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(model_cfg).to(self.device)
        self.model = torch.compile(self.model)
        self.train_cfg = train_cfg

    def _compute_loss(self, pred, target, forces):
        """Compute weighted loss with optional relative normalization."""
        if getattr(self.train_cfg, "loss_type", "mse") == "relative_mse":
            eps = getattr(self.train_cfg, "relative_loss_eps", 1e-6)
            target_norm = target.norm(dim=-1, keepdim=True).clamp(min=eps)
            per_particle = ((pred - target) / target_norm).pow(2).sum(dim=-1)
        else:
            per_particle = (pred - target).pow(2).sum(dim=-1)

        passive_w = getattr(self.train_cfg, "passive_weight", 1.0)
        if passive_w > 1.0:
            force_norm = forces.norm(dim=-1)
            weights = torch.where(
                force_norm < 1e-6,
                torch.full_like(force_norm, passive_w),
                torch.ones_like(force_norm),
            )
            return (weights * per_particle).mean()

        return per_particle.mean()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        forces = batch.forces
        loss = self._compute_loss(pred, batch.y, forces)

        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("lr", self.optimizers().param_groups[0]["lr"], batch_size=batch.num_graphs)

        with torch.no_grad():
            mse = (pred - batch.y).pow(2).mean()
            self.log("train_mse", mse, batch_size=batch.num_graphs)

        return loss

    # ── Validation with per-category metrics ──────────────────────────

    def on_validation_epoch_start(self):
        self._val_preds = []
        self._val_targets = []
        self._val_forces = []
        self._val_geo_ids = []
        self._val_num_nodes = []

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        target = batch.y
        forces = batch.forces
        loss = self._compute_loss(pred, target, forces)

        # Accumulate for epoch-end per-category metrics
        self._val_preds.append(pred.detach())
        self._val_targets.append(target.detach())
        self._val_forces.append(forces.detach())
        # Expand per-graph attributes to per-node
        self._val_geo_ids.append(batch.geometry_id[batch.batch].detach())
        nn_per_graph = batch.ptr[1:] - batch.ptr[:-1]
        self._val_num_nodes.append(nn_per_graph[batch.batch].detach())

        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)

        return loss

    def on_validation_epoch_end(self):
        pred = torch.cat(self._val_preds)
        target = torch.cat(self._val_targets)
        forces = torch.cat(self._val_forces)
        geo_id = torch.cat(self._val_geo_ids)
        num_nodes = torch.cat(self._val_num_nodes)

        # Core metrics
        error_norm = (pred - target).norm(dim=-1)
        true_norm = target.norm(dim=-1).clamp(min=1e-12)
        rel_error = error_norm / true_norm
        abs_error = error_norm
        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        cos_mask = true_norm > 0.01
        abs_err_components = (pred - target).abs()

        # Overall metrics
        self.log("val_mse", (pred - target).pow(2).mean())
        self.log("val_rel_error", rel_error.mean(), prog_bar=True)
        self.log("val_mae_x", abs_err_components[:, 0].mean())
        self.log("val_mae_y", abs_err_components[:, 1].mean())
        self.log("val_mae_z", abs_err_components[:, 2].mean())
        if cos_mask.any():
            self.log("val_cosine_sim", cos_sim[cos_mask].mean())

        # Active vs passive (|F| > 0.1)
        force_norm = forces.norm(dim=-1)
        active = force_norm > 0.1
        passive = ~active
        if active.any():
            self.log("val_rel_error_active", rel_error[active].mean())
            self.log("val_abs_error_active", abs_error[active].mean())
        if passive.any():
            self.log("val_rel_error_passive", rel_error[passive].mean())
            self.log("val_abs_error_passive", abs_error[passive].mean())

        # Per geometry
        for gid, gname in [(0, "nbody"), (1, "pse")]:
            mask = geo_id == gid
            if mask.any():
                self.log(f"val_rel_error_{gname}", rel_error[mask].mean())
                self.log(f"val_abs_error_{gname}", abs_error[mask].mean())

        # Per particle count
        for n in [16, 32, 64]:
            mask = num_nodes == n
            if mask.any():
                self.log(f"val_rel_error_N{n}", rel_error[mask].mean())

        # Histograms (TensorBoard Histograms tab)
        if self.logger and hasattr(self.logger, "experiment"):
            writer = self.logger.experiment
            step = self.current_epoch
            writer.add_histogram("val_relative_error_dist", rel_error.cpu(), step)
            writer.add_histogram("val_absolute_error_dist", abs_error.cpu(), step)
            if cos_mask.any():
                writer.add_histogram("val_cosine_sim_dist", cos_sim[cos_mask].cpu(), step)

        # Clear accumulators
        del self._val_preds, self._val_targets, self._val_forces
        del self._val_geo_ids, self._val_num_nodes

    # ── Test step ─────────────────────────────────────────────────────

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.model.eval()
        pred = self.model(batch)
        target = batch.y

        mse = (pred - target).pow(2).mean()
        mae = (pred - target).abs().mean()
        rel_error = ((pred - target).norm(dim=-1)
                     / target.norm(dim=-1).clamp(min=1e-12)).mean()

        self.log("test_mse", mse, batch_size=batch.num_graphs)
        self.log("test_mae", mae, batch_size=batch.num_graphs)
        self.log("test_rel_error", rel_error, batch_size=batch.num_graphs)

    # ── Optimizer ─────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )

        warmup_epochs = getattr(self.train_cfg, "warmup_epochs", 0)
        plateau = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
        )

        if warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            scheduler = SequentialLR(optimizer, [warmup, plateau],
                                    milestones=[warmup_epochs])
        else:
            scheduler = plateau

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
