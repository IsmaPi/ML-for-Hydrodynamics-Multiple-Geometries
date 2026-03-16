#!/usr/bin/env python3
"""CLI for training with PyTorch Lightning.

Usage:
    python scripts/train.py --data-config default --model-config torchmd_et
    python scripts/train.py --data-config default --model-config torchmd_gn --train-config default
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

from utils.config import (
    load_yaml, build_data_config, build_model_config,
    build_training_config, resolve_config_path,
)
from utils.seed import set_seed
from data.dataset import HydroDataModule
from models.lightning_wrapper import HydroLitModule


def make_run_name(model_cfg) -> str:
    """Generate enumerated run name: {model_type}/run_001."""
    base = model_cfg.model_type
    parent = Path("saved_models") / base
    if not parent.exists():
        return f"{base}/run_001"
    existing = sorted(parent.glob("run_*"))
    nums = []
    for d in existing:
        try:
            nums.append(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(nums, default=0) + 1
    return f"{base}/run_{next_num:03d}"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Train hydrodynamic displacement model")
    parser.add_argument("--data-config", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--train-config", type=str, default="configs/training/default.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    # Resolve short config names
    data_path = resolve_config_path(args.data_config, "data")
    model_path = resolve_config_path(args.model_config, "model")
    train_path = resolve_config_path(args.train_config, "training")

    data_cfg = build_data_config(load_yaml(data_path))
    model_cfg = build_model_config(load_yaml(model_path))
    train_cfg = build_training_config(load_yaml(train_path))

    run_name = args.run_name or make_run_name(model_cfg)
    set_seed(train_cfg.seed)

    log.info("Run: %s", run_name)
    log.info("Model: %s | Epochs: %d", model_cfg.model_type, train_cfg.max_epochs)

    # Data
    datamodule = HydroDataModule(
        data_dir=data_cfg.output_dir,
        geometries=data_cfg.geometries,
        num_particles=data_cfg.num_particles,
        batch_size=train_cfg.batch_size,
        seed=train_cfg.seed,
    )

    # Model
    lit_model = HydroLitModule(model_cfg, train_cfg)
    total_params = sum(p.numel() for p in lit_model.parameters())
    log.info("Model parameters: %s", f"{total_params:,}")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg.max_epochs,
        accelerator="auto",
        precision=train_cfg.precision,
        gradient_clip_val=train_cfg.gradient_clip,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=train_cfg.patience, mode="min"),
            ModelCheckpoint(
                dirpath=f"saved_models/{run_name}",
                monitor="val_loss",
                save_top_k=1,
                filename="best",
            ),
        ],
        logger=TensorBoardLogger("logs/", name=run_name),
        log_every_n_steps=10,
    )

    trainer.fit(lit_model, datamodule)

    log.info("Training complete.")
    log.info("Best model: saved_models/%s/best.ckpt", run_name)
    log.info("Logs: logs/%s/", run_name)


if __name__ == "__main__":
    main()
