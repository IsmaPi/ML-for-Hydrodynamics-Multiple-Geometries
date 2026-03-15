#!/usr/bin/env python3
"""CLI for model evaluation.

Usage:
    python scripts/evaluate.py --checkpoint saved_models/torchmd_et/run_001/best.ckpt \
        --eval-mode single_step --data-config default

    python scripts/evaluate.py --checkpoint saved_models/torchmd_et/run_001/best.ckpt \
        --eval-mode pair_sweep
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.config import load_yaml, build_data_config, resolve_config_path
from data.dataset import HydroDataModule
from models.lightning_wrapper import HydroLitModule
from evaluation.single_step import evaluate_single_step
from evaluation.pair_sweep import evaluate_pair_sweep, plot_pair_sweep


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Evaluate hydrodynamic model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--eval-mode", type=str, required=True,
                        choices=["single_step", "pair_sweep"])
    parser.add_argument("--data-config", type=str, default=None,
                        help="Data config (required for single_step)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from checkpoint
    lit_model = HydroLitModule.load_from_checkpoint(args.checkpoint)
    lit_model.eval()
    lit_model.to(device)
    model = lit_model.model

    if args.eval_mode == "single_step":
        if args.data_config is None:
            log.error("--data-config required for single_step mode")
            sys.exit(1)

        data_path = resolve_config_path(args.data_config, "data")
        data_cfg = build_data_config(load_yaml(data_path))

        datamodule = HydroDataModule(
            data_dir=data_cfg.output_dir,
            geometries=data_cfg.geometries,
            num_particles=data_cfg.num_particles,
        )
        datamodule.setup()

        results = evaluate_single_step(model, datamodule.val_dataset, device)
        log.info("Single-step results: MSE=%.6e MAE=%.6e RelErr=%.4f",
                 results["mse"], results["mae"], results["relative_error"])

        output_path = args.output or "results/eval_single_step.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Results saved to %s", output_path)

    elif args.eval_mode == "pair_sweep":
        results = evaluate_pair_sweep(model, device, geometry="nbody_open")

        output_path = args.output or "results/pair_sweep.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        plot_pair_sweep(results, save_path=output_path, particle_idx=0)
        plot_pair_sweep(
            results,
            save_path=str(Path(output_path).with_name("pair_sweep_p2.png")),
            particle_idx=1,
        )

        log.info("Pair sweep plots saved to %s", Path(output_path).parent)


if __name__ == "__main__":
    main()
