"""PyTorch Lightning module wrapping the hydrodynamic displacement models.

Handles training/validation step, loss computation, optimizer/scheduler setup.
Works with both HydroTorchMD_ET and HydroTorchMD_GN.
"""

import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    """Lightning module for hydrodynamic displacement prediction.

    Wraps either HydroTorchMD_ET or HydroTorchMD_GN and handles
    the training loop, validation, and optimizer configuration.
    """

    def __init__(self, model_cfg, train_cfg):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(model_cfg)
        self.train_cfg = train_cfg

        # Analytical self-mobility baseline for comparison
        viscosity = getattr(model_cfg, "viscosity", 1.0)
        a = getattr(model_cfg, "hydrodynamic_radius", 1.0)
        self.self_mobility = 1.0 / (6.0 * math.pi * viscosity * a)

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
        forces = batch.x[:, :3]
        loss = self._compute_loss(pred, batch.y, forces)

        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("lr", self.optimizers().param_groups[0]["lr"], batch_size=batch.num_graphs)

        with torch.no_grad():
            baseline = self.self_mobility * forces
            correction = pred - baseline
            self.log("train_correction_rms", correction.pow(2).mean().sqrt(),
                     batch_size=batch.num_graphs)
            mse = (pred - batch.y).pow(2).mean()
            self.log("train_mse", mse, batch_size=batch.num_graphs)

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        target = batch.y
        forces = batch.x[:, :3]
        loss = self._compute_loss(pred, target, forces)

        with torch.no_grad():
            # Baseline: pure self-mobility (no interactions)
            baseline = self.self_mobility * forces
            baseline_mse = (baseline - target).pow(2).mean()
            pred_mse = (pred - target).pow(2).mean()

            # Per-particle relative error
            true_norm = target.norm(dim=-1).clamp(min=1e-12)
            rel_error = (pred - target).norm(dim=-1) / true_norm

            # Per-component MAE
            abs_err = (pred - target).abs()

            # Improvement over baseline (using raw MSE, not weighted loss)
            improvement = 1.0 - pred_mse / baseline_mse.clamp(min=1e-12)

            # Correction magnitude
            correction = pred - baseline

        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("val_mse", pred_mse, batch_size=batch.num_graphs)
        self.log("val_baseline_mse", baseline_mse, batch_size=batch.num_graphs)
        self.log("val_improvement", improvement, prog_bar=True, batch_size=batch.num_graphs)
        self.log("val_rel_error", rel_error.mean(), batch_size=batch.num_graphs)
        self.log("val_mae_x", abs_err[:, 0].mean(), batch_size=batch.num_graphs)
        self.log("val_mae_y", abs_err[:, 1].mean(), batch_size=batch.num_graphs)
        self.log("val_mae_z", abs_err[:, 2].mean(), batch_size=batch.num_graphs)
        self.log("val_correction_rms", correction.pow(2).mean().sqrt(),
                 batch_size=batch.num_graphs)

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """Evaluate on test set. Reports raw MSE, MAE, and relative error."""
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

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
