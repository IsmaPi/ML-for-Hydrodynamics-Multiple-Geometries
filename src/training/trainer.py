"""Main training loop supporting single, mixed, and leave-one-out strategies.

Validation is always decomposed per geometry to track generalization,
even when training on mixed-geometry data.

Saves:
  - Training history (CSV) to results/<run_name>/history.csv
  - Best model to saved_models/<run_name>/best.pt
  - Periodic checkpoints to saved_models/<run_name>/
"""

import csv
import json
import logging
import time
import torch
from pathlib import Path

log = logging.getLogger(__name__)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader

from .losses import DisplacementMSELoss
from ..utils.logging import Logger


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model + optimizer state."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    """Load model + optimizer state."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt.get("metrics", {})


class Trainer:
    """Training loop with per-geometry validation tracking.

    Saves outputs to:
      - saved_models/<run_name>/best.pt   (best model weights)
      - results/<run_name>/history.csv    (per-epoch train/val losses)
      - results/<run_name>/summary.json   (final metrics summary)
      - saved_models/<run_name>/epoch_NNNN.pt  (periodic checkpoints)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_datasets: dict,
        config,
        logger: Logger,
        run_name: str = "default",
    ):
        self.config = config
        self.logger = logger
        self.run_name = run_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        self.model = model.to(self.device)

        log.info("Device: %s%s", self.device, f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else "")
        if self.use_amp:
            log.info("Mixed precision (AMP): enabled")

        # Output directories
        self.model_dir = Path("saved_models") / run_name
        self.results_dir = Path("results") / run_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Training history: list of dicts, one per recorded epoch
        self.history = []

        # Use multiple workers when data is pre-cached for faster loading
        num_workers = 4 if self.device.type == "cuda" else 0

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
        )

        # Per-geometry validation loaders
        self.val_loaders = {
            geo: DataLoader(
                ds, batch_size=config.batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=self.device.type == "cuda",
                persistent_workers=num_workers > 0,
            )
            for geo, ds in val_datasets.items()
        }

        self.criterion = DisplacementMSELoss()

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Mixed precision scaler (only used on CUDA)
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.num_epochs
            )
        elif config.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, patience=10, factor=0.5
            )
        else:
            self.scheduler = None

        self.best_val_loss = float("inf")
        self.patience = config.early_stopping_patience
        self.epochs_without_improvement = 0

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0

        for batch_data in self.train_loader:
            batch_data = batch_data.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                pred = self.model(batch_data)
                loss = self.criterion(pred, batch_data.y, batch_data.batch)

            self.scaler.scale(loss).backward()

            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> dict:
        """Evaluate on each geometry's validation set separately.

        Returns dict: {geometry_key: loss, ..., "avg": average_loss}
        """
        self.model.eval()
        results = {}

        for geo, loader in self.val_loaders.items():
            total_loss = 0.0
            num_batches = 0
            for batch_data in loader:
                batch_data = batch_data.to(self.device, non_blocking=True)
                with autocast("cuda", enabled=self.use_amp):
                    pred = self.model(batch_data)
                    loss = self.criterion(pred, batch_data.y, batch_data.batch)
                total_loss += loss.item()
                num_batches += 1
            if num_batches > 0:
                results[geo] = total_loss / num_batches

        if results:
            results["avg"] = sum(results.values()) / len(results)

        return results

    def fit(self):
        """Main training loop."""
        val_results = {}
        t_start = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            t_epoch = time.time()
            train_loss = self.train_epoch(epoch)
            epoch_time = time.time() - t_epoch

            # Logging
            if epoch % self.config.log_every == 0:
                self.logger.log_scalar("train/loss", train_loss, epoch)
            log.info("Epoch %d/%d | train_loss=%.6f | %.1fs", epoch, self.config.num_epochs, train_loss, epoch_time)

            # Validation
            if epoch % self.config.eval_every == 0:
                val_results = self.validate()
                for geo, val_loss in val_results.items():
                    self.logger.log_scalar(f"val/{geo}", val_loss, epoch)
                log.info("  val: %s", " | ".join(f"{k}={v:.6f}" for k, v in val_results.items()))

                # Save best model to saved_models/
                avg_val = val_results.get("avg", float("inf"))
                if avg_val < self.best_val_loss:
                    self.best_val_loss = avg_val
                    self.epochs_without_improvement = 0
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        val_results, str(self.model_dir / "best.pt"),
                    )
                else:
                    self.epochs_without_improvement += self.config.eval_every

                # Early stopping
                if self.patience > 0 and self.epochs_without_improvement >= self.patience:
                    log.info("Early stopping at epoch %d (no val improvement for %d epochs)", epoch, self.epochs_without_improvement)
                    self._save_history()
                    self._save_summary(val_results)
                    total_time = time.time() - t_start
                    log.info("Training complete in %.1fs. Best val loss: %.6f", total_time, self.best_val_loss)
                    log.info("Model saved to:   %s", self.model_dir / "best.pt")
                    log.info("Metrics saved to: %s/", self.results_dir)
                    return

            # Record history row
            row = {"epoch": epoch, "train_loss": train_loss}
            for k, v in val_results.items():
                row[f"val_{k}"] = v
            self.history.append(row)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    val_avg = val_results.get("avg", train_loss) if epoch % self.config.eval_every == 0 else train_loss
                    self.scheduler.step(val_avg)
                else:
                    self.scheduler.step()

            # Periodic checkpoint
            if epoch % self.config.checkpoint_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    {}, str(self.model_dir / f"epoch_{epoch:04d}.pt"),
                )

        # Save training history CSV
        self._save_history()
        # Save final summary JSON
        self._save_summary(val_results)

        total_time = time.time() - t_start
        log.info("Training complete in %.1fs. Best val loss: %.6f", total_time, self.best_val_loss)
        log.info("Model saved to:   %s", self.model_dir / "best.pt")
        log.info("Metrics saved to: %s/", self.results_dir)

    def _save_history(self):
        """Write training history to CSV."""
        if not self.history:
            return
        csv_path = self.results_dir / "history.csv"
        fieldnames = list(self.history[0].keys())
        # Collect all possible column names across all rows
        for row in self.history:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

    def _save_summary(self, final_val_results: dict):
        """Write final training summary to JSON."""
        summary = {
            "run_name": self.run_name,
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.config.num_epochs,
            "final_val_results": {k: float(v) for k, v in final_val_results.items()},
        }
        json_path = self.results_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
