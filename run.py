#!/usr/bin/env python3
"""Main script to run the full pipeline: generate data -> train -> evaluate.

Usage:
    # Full pipeline with synthetic data (no libMobility needed)
    python run.py --synthetic

    # Full pipeline with libMobility (requires CUDA GPU + libMobility installed)
    python run.py

    # Only specific stages
    python run.py --stage generate --synthetic
    python run.py --stage train --model gnn
    python run.py --stage evaluate --model gnn

    # Custom configs
    python run.py --data-config configs/data/mixed_all.yaml \
                  --model gnn \
                  --train-config configs/training/leave_one_out.yaml
"""

import argparse
import sys
import subprocess
from pathlib import Path


def _find_latest_run(base: str):
    """Find the latest run_NNN under saved_models/<base>/.

    Returns (run_name, checkpoint_path) or (None, None) if not found.
    Also checks the old flat layout saved_models/<base>/best.pt as fallback.
    """
    parent = Path("saved_models") / base

    # Check enumerated runs first
    if parent.exists():
        run_dirs = sorted(parent.glob("run_*"), reverse=True)
        for d in run_dirs:
            ckpt = d / "best.pt"
            if ckpt.exists():
                return f"{base}/{d.name}", str(ckpt)

        # Fallback: flat layout (pre-enumeration runs)
        flat_ckpt = parent / "best.pt"
        if flat_ckpt.exists():
            return base, str(flat_ckpt)

    return None, None


def run_cmd(cmd: list, description: str):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        print(f"\nFailed: {description}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full ML-for-Hydrodynamics pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["all", "generate", "train", "evaluate"],
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--model", type=str, default="torchmd_gn",
        choices=["gnn", "set_transformer", "torchmd_gn", "torchmd_et", "both"],
        help="Which model to train/evaluate (default: torchmd_gn)",
    )
    parser.add_argument(
        "--data-config", type=str, default="configs/data/nbody_open.yaml",
        help="Data config YAML (default: configs/data/nbody_open.yaml)",
    )
    parser.add_argument(
        "--train-config", type=str, default="configs/training/single_geometry.yaml",
        help="Training config YAML (default: configs/training/single_geometry.yaml)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (no libMobility needed)",
    )
    parser.add_argument(
        "--num-particles", type=int, default=None,
        help="Override particle count (single N value)",
    )
    args = parser.parse_args()

    python = sys.executable
    models = ["torchmd_gn", "torchmd_et"] if args.model == "both" else [args.model]

    # ----------------------------------------------------------------
    # Stage 1: Generate data
    # ----------------------------------------------------------------
    if args.stage in ("all", "generate"):
        cmd = [python, "scripts/generate_data.py", "--config", args.data_config]
        if args.synthetic:
            cmd.append("--synthetic")
        if args.num_particles:
            cmd.extend(["--num-particles", str(args.num_particles)])
        run_cmd(cmd, "Generating training data")

    # Derive strategy name from train config filename
    strategy = Path(args.train_config).stem  # e.g. "single_geometry"

    # ----------------------------------------------------------------
    # Stage 2: Train
    # ----------------------------------------------------------------
    if args.stage in ("all", "train"):
        for model_name in models:
            model_config = f"configs/model/{model_name}.yaml"
            # Let train.py auto-enumerate the run name (run_001, run_002, ...)
            cmd = [
                python, "scripts/train.py",
                "--data-config", args.data_config,
                "--model-config", model_config,
                "--train-config", args.train_config,
            ]
            run_cmd(cmd, f"Training {model_name} ({strategy})")

    # ----------------------------------------------------------------
    # Stage 3: Evaluate
    # ----------------------------------------------------------------
    if args.stage in ("all", "evaluate"):
        for model_name in models:
            model_config = f"configs/model/{model_name}.yaml"
            base = f"{model_name}_{strategy}"

            # Find the latest enumerated run
            run_name, checkpoint = _find_latest_run(base)
            if checkpoint is None:
                print(f"Warning: no checkpoint found for {base}, skipping evaluation.")
                continue

            # Single-step evaluation
            cmd = [
                python, "scripts/evaluate.py",
                "--data-config", args.data_config,
                "--model-config", model_config,
                "--checkpoint", checkpoint,
                "--eval-mode", "single_step",
                "--run-name", run_name,
            ]
            run_cmd(cmd, f"Evaluating {model_name} (single-step)")

            # Scaling evaluation
            cmd = [
                python, "scripts/evaluate.py",
                "--model-config", model_config,
                "--checkpoint", checkpoint,
                "--eval-mode", "scaling",
                "--run-name", run_name,
            ]
            run_cmd(cmd, f"Evaluating {model_name} (scaling)")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
