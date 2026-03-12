#!/usr/bin/env python3
"""Main script to run the full pipeline: generate data -> train -> evaluate.

Usage:
    # Full pipeline with synthetic data (no libMobility needed)
    python run.py --synthetic

    # Full pipeline with libMobility (requires CUDA GPU + libMobility installed)
    python run.py

    # Only specific stages
    python run.py --stage generate --synthetic
    python run.py --stage train --model torchmd_gn
    python run.py --stage evaluate --model torchmd_gn

    # Custom configs
    python run.py --data-config configs/data/mixed_all.yaml \
                  --model torchmd_gn \
                  --train-config configs/training/leave_one_out.yaml
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _find_latest_run(base: str):
    """Find the latest run_NNN under saved_models/<base>/.

    Returns (run_name, checkpoint_path) or (None, None) if not found.
    """
    parent = Path("saved_models") / base

    if parent.exists():
        run_dirs = sorted(parent.glob("run_*"), reverse=True)
        for d in run_dirs:
            ckpt = d / "best.pt"
            if ckpt.exists():
                return f"{base}/{d.name}", str(ckpt)

    return None, None


def run_cmd(cmd: list, description: str):
    """Run a command and log its output."""
    log.info("\n%s\n  %s\n%s\n", "=" * 60, description, "=" * 60)
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        log.error("Failed: %s", description)
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
        choices=["torchmd_gn", "torchmd_et", "both"],
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
    parser.add_argument(
        "--leave-out-geometry", type=str, default=None,
        help="Override leave_out_geometry in training config (e.g. nbody_open)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve short config names (e.g. "mixed_all" -> "configs/data/mixed_all.yaml")
    sys.path.insert(0, str(Path(__file__).parent))
    from src.utils.config import resolve_config_path
    args.data_config = resolve_config_path(args.data_config, "data")
    args.train_config = resolve_config_path(args.train_config, "training")

    # Resolve short config names (e.g. "mixed_all" -> "configs/data/mixed_all.yaml")
    sys.path.insert(0, str(Path(__file__).parent))
    from src.utils.config import resolve_config_path
    args.data_config = resolve_config_path(args.data_config, "data")
    args.train_config = resolve_config_path(args.train_config, "training")

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

    # Derive strategy from the YAML content to match train.py's make_run_name()
    # train.py uses train_cfg.strategy (e.g. "single", "mixed", "leave_one_out")
    import yaml
    with open(args.train_config) as _f:
        _train_raw = yaml.safe_load(_f) or {}
    strategy = _train_raw.get("strategy", Path(args.train_config).stem)

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
            if args.leave_out_geometry:
                cmd.extend(["--leave-out-geometry", args.leave_out_geometry])
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
                log.warning("No checkpoint found for %s, skipping evaluation.", base)
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

    log.info("\n%s\n  Pipeline complete!\n%s", "=" * 60, "=" * 60)


if __name__ == "__main__":
    main()
