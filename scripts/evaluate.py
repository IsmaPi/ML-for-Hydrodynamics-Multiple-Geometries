#!/usr/bin/env python3
"""CLI entry point for evaluation.

Usage:
    python scripts/evaluate.py \
        --data-config configs/data/mixed_all.yaml \
        --model-config configs/model/torchmd_gn.yaml \
        --checkpoint saved_models/torchmd_gn_single_geometry/run_001/best.pt \
        --eval-mode single_step

    python scripts/evaluate.py \
        --data-config configs/data/mixed_all.yaml \
        --model-config configs/model/torchmd_et.yaml \
        --checkpoint saved_models/torchmd_et_single_geometry/run_001/best.pt \
        --eval-mode generalization

    python scripts/evaluate.py \
        --model-config configs/model/torchmd_gn.yaml \
        --checkpoint saved_models/torchmd_gn_single_geometry/run_001/best.pt \
        --eval-mode scaling
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.utils.config import load_yaml, merge_configs, build_data_config, build_model_config
from src.utils.seed import set_seed
from src.data.dataset import HydrodynamicsDataset
from src.data.normalization import FeatureNormalizer
from src.models.torchmd_gn import HydroTorchMD_GN
from src.models.torchmd_et import HydroTorchMD_ET
from src.training.trainer import load_checkpoint
from src.evaluation.single_step import evaluate_single_step
from src.evaluation.scaling import profile_scaling
from src.evaluation.generalization import evaluate_cross_geometry, print_generalization_report
from src.evaluation.rollout import evaluate_rollout


def load_model(model_cfg, checkpoint_path, device):
    """Load trained model from checkpoint."""
    if model_cfg.model_type == "torchmd_gn":
        model = HydroTorchMD_GN(model_cfg)
    elif model_cfg.model_type == "torchmd_et":
        model = HydroTorchMD_ET(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_cfg.model_type}")

    load_checkpoint(model, None, checkpoint_path, device)
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate hydrodynamics ML model")
    parser.add_argument("--data-config", type=str, default=None)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (e.g. saved_models/torchmd_gn_single/run_001/best.pt)")
    parser.add_argument("--eval-mode", type=str, required=True,
                        choices=["single_step", "generalization", "scaling", "rollout"])
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON (default: results/<run_name>/eval_<mode>.json)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for auto-saving results (derived from checkpoint path if omitted)")
    parser.add_argument("--rollout-steps", type=int, default=50,
                        help="Number of rollout steps (default: 50)")
    parser.add_argument("--rollout-n", type=int, default=64,
                        help="Number of particles for rollout (default: 64)")
    parser.add_argument("--rollout-dt", type=float, default=0.01,
                        help="Timestep for rollout (default: 0.01)")
    args = parser.parse_args()

    # Derive run_name from checkpoint path if not given: saved_models/<run_name>/best.pt
    if args.run_name is None:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.parent.parent.name == "saved_models":
            args.run_name = ckpt_path.parent.name
        else:
            args.run_name = "eval"

    # Auto-set output path if not provided
    if args.output is None:
        results_dir = Path("results") / args.run_name
        results_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(results_dir / f"eval_{args.eval_mode}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    raw_model = load_yaml(args.model_config)
    model_cfg = build_model_config(raw_model)
    model = load_model(model_cfg, args.checkpoint, device)

    results = {}

    if args.eval_mode == "scaling":
        print("Profiling scaling...")
        results = profile_scaling(
            model, model_cfg.input_dim, device,
            model_type=model_cfg.model_type,
            cutoff_radius=getattr(model_cfg, "cutoff_radius", 30.0),
        )

    elif args.eval_mode in ("single_step", "generalization"):
        if args.data_config is None:
            print("Error: --data-config required for single_step/generalization modes.")
            sys.exit(1)

        raw_data = load_yaml(args.data_config)
        data_cfg = build_data_config(raw_data)

        # Load normalizer: z-score stats (mean, std) computed on training data
        # during train.py. Used to scale inputs (forces, theta) and denormalize
        # predicted displacements back to physical units.
        normalizer_path = "data/processed/normalizer.pt"
        normalizer = None
        if Path(normalizer_path).exists():
            normalizer = FeatureNormalizer()
            normalizer.load(normalizer_path)

        # Build per-geometry datasets
        geo_datasets = {}
        for geo in data_cfg.geometries:
            ds = HydrodynamicsDataset(
                data_dir=data_cfg.output_dir,
                geometry_keys=[geo],
                particle_counts=data_cfg.num_particles,
                normalizer=normalizer,
            )
            if len(ds) > 0:
                geo_datasets[geo] = ds

        if args.eval_mode == "single_step":
            for geo, ds in geo_datasets.items():
                metrics = evaluate_single_step(model, ds, normalizer, device)
                print(f"{geo}: MSE={metrics['mse']:.6e} MAE={metrics['mae']:.6e} RelErr={metrics['relative_error']:.4f}")
                results[geo] = metrics

        elif args.eval_mode == "generalization":
            results = evaluate_cross_geometry(model, geo_datasets, normalizer, device)
            print_generalization_report(results)

    elif args.eval_mode == "rollout":
        if args.data_config is None:
            print("Error: --data-config required for rollout mode.")
            sys.exit(1)

        raw_data = load_yaml(args.data_config)
        data_cfg = build_data_config(raw_data)

        # Load normalizer (see comment above for single_step/generalization)
        normalizer_path = "data/processed/normalizer.pt"
        normalizer = None
        if Path(normalizer_path).exists():
            normalizer = FeatureNormalizer()
            normalizer.load(normalizer_path)

        all_trajectories = {}
        for geo in data_cfg.geometries:
            print(f"Rollout evaluation for {geo}...")
            rollout_result = evaluate_rollout(
                model=model,
                geometry_key=geo,
                config=raw_data,
                normalizer=normalizer,
                N=args.rollout_n,
                num_steps=args.rollout_steps,
                dt=args.rollout_dt,
                device=device,
                model_type=model_cfg.model_type,
                cutoff_radius=getattr(model_cfg, "cutoff_radius", 30.0),
            )

            results[geo] = {
                "per_step_error": rollout_result["per_step_error"].tolist(),
                "cumulative_error": rollout_result["cumulative_error"].tolist(),
            }
            all_trajectories[geo] = {
                "pred": rollout_result["pred_trajectory"],
                "true": rollout_result["true_trajectory"],
            }

        # Save trajectories as npz
        results_dir = Path(args.output).parent
        traj_data = {}
        for geo, trajs in all_trajectories.items():
            traj_data[f"{geo}_pred"] = trajs["pred"]
            traj_data[f"{geo}_true"] = trajs["true"]
        np.savez(str(results_dir / "eval_rollout_trajectories.npz"), **traj_data)
        print(f"Trajectories saved to {results_dir / 'eval_rollout_trajectories.npz'}")

    # Save results
    if args.output:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        with open(args.output, "w") as f:
            json.dump(convert(results), f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
