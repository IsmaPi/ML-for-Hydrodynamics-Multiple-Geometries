#!/usr/bin/env python3
"""Main pipeline: generate data -> train model -> evaluate.

Usage:
    python run.py --stage all --model torchmd_et
    python run.py --stage generate --data-config default
    python run.py --stage train --model torchmd_et
    python run.py --stage evaluate --model torchmd_et
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

import torch

from utils.config import (
    load_yaml, build_data_config, build_model_config,
    build_training_config, resolve_config_path,
)
from utils.seed import set_seed
from data.generate import generate_all
from data.dataset import HydroDataModule
from models.lightning_wrapper import HydroLitModule
from evaluation.single_step import evaluate_single_step
from evaluation.pair_sweep import (
    evaluate_pair_sweep, plot_pair_sweep,
    plot_pair_sweep_both_particles, plot_pair_sweep_error,
)
from evaluation.plot_training import plot_training_curves

log = logging.getLogger(__name__)


def _make_run_name(model_type: str) -> str:
    """Generate enumerated run name."""
    parent = Path("saved_models") / model_type
    if not parent.exists():
        return f"{model_type}/run_001"
    existing = sorted(parent.glob("run_*"))
    nums = []
    for d in existing:
        try:
            nums.append(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(nums, default=0) + 1
    return f"{model_type}/run_{next_num:03d}"


def _find_latest_checkpoint(model_type: str) -> str:
    """Find the latest best.ckpt for a model type."""
    parent = Path("saved_models") / model_type
    if not parent.exists():
        raise FileNotFoundError(f"No saved models found for {model_type}")
    runs = sorted(parent.glob("run_*/best.ckpt"))
    if not runs:
        raise FileNotFoundError(f"No checkpoints found in {parent}")
    return str(runs[-1])


def run_generate(data_cfg):
    """Generate training data."""
    log.info("=== Generating data ===")
    generate_all(data_cfg)
    log.info("Data generation complete.")


def run_train(model_type: str, data_cfg, model_cfg, train_cfg):
    """Train a model."""
    run_name = _make_run_name(model_type)
    set_seed(train_cfg.seed)

    log.info("=== Training %s (%s) ===", model_type, run_name)

    datamodule = HydroDataModule(
        data_dir=data_cfg.output_dir,
        geometries=data_cfg.geometries,
        num_particles=data_cfg.num_particles,
        batch_size=train_cfg.batch_size,
        seed=train_cfg.seed,
    )

    lit_model = HydroLitModule(model_cfg, train_cfg)
    total_params = sum(p.numel() for p in lit_model.parameters())
    log.info("Model parameters: %s", f"{total_params:,}")

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
    log.info("Training complete. Best model: saved_models/%s/best.ckpt", run_name)
    return run_name


def run_evaluate(model_type: str, data_cfg):
    """Evaluate the latest checkpoint."""
    log.info("=== Evaluating %s ===", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = _find_latest_checkpoint(model_type)
    lit_model = HydroLitModule.load_from_checkpoint(checkpoint)
    lit_model.eval()
    lit_model.to(device)
    model = lit_model.model

    # Single-step evaluation
    datamodule = HydroDataModule(
        data_dir=data_cfg.output_dir,
        geometries=data_cfg.geometries,
        num_particles=data_cfg.num_particles,
    )
    datamodule.setup()

    results = evaluate_single_step(model, datamodule.val_dataset, device)
    log.info("Single-step: MSE=%.6e MAE=%.6e RelErr=%.4f",
             results["mse"], results["mae"], results["relative_error"])

    # Pair sweep evaluation
    pair_results = evaluate_pair_sweep(model, device, geometry="nbody_open")
    results_dir = Path("results") / model_type
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_pair_sweep(pair_results, save_path=str(results_dir / "pair_sweep_p1.png"), particle_idx=0)
    plot_pair_sweep(pair_results, save_path=str(results_dir / "pair_sweep_p2.png"), particle_idx=1)
    plot_pair_sweep_both_particles(pair_results, save_path=str(results_dir / "pair_sweep_both.png"))
    plot_pair_sweep_error(pair_results, save_path=str(results_dir / "pair_sweep_error.png"))

    # Training curves from TensorBoard logs
    try:
        plot_training_curves(model_type, save_dir=str(results_dir))
    except FileNotFoundError as e:
        log.warning("Could not plot training curves: %s", e)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="ML for Hydrodynamics pipeline")
    parser.add_argument("--stage", choices=["generate", "train", "evaluate", "all"],
                        default="all")
    parser.add_argument("--data-config", type=str, default="configs/data/default.yaml")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Model config (auto-resolved from --model if not given)")
    parser.add_argument("--train-config", type=str, default="configs/training/default.yaml")
    parser.add_argument("--model", choices=["torchmd_gn", "torchmd_et", "both"],
                        default="torchmd_et")
    args = parser.parse_args()

    # Resolve configs
    data_path = resolve_config_path(args.data_config, "data")
    train_path = resolve_config_path(args.train_config, "training")
    data_cfg = build_data_config(load_yaml(data_path))
    train_cfg = build_training_config(load_yaml(train_path))

    # Determine which models to run
    model_types = ["torchmd_gn", "torchmd_et"] if args.model == "both" else [args.model]

    if args.stage in ("generate", "all"):
        run_generate(data_cfg)

    for model_type in model_types:
        # Resolve model config
        if args.model_config:
            model_path = resolve_config_path(args.model_config, "model")
        else:
            model_path = resolve_config_path(model_type, "model")
        model_cfg = build_model_config(load_yaml(model_path))

        if args.stage in ("train", "all"):
            run_train(model_type, data_cfg, model_cfg, train_cfg)

        if args.stage in ("evaluate", "all"):
            run_evaluate(model_type, data_cfg)


if __name__ == "__main__":
    main()
