#!/usr/bin/env python3
"""CLI entry point for data generation.

Usage:
    python scripts/generate_data.py --config configs/data/mixed_all.yaml
    python scripts/generate_data.py --config configs/data/nbody_open.yaml --synthetic
    python scripts/generate_data.py --config configs/data/mixed_all.yaml --geometry nbody_open --num-particles 32
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generate import generate_dataset_for_geometry
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory data for hydrodynamics ML")
    parser.add_argument("--config", type=str, required=True, help="Path to data config YAML")
    parser.add_argument("--geometry", type=str, default=None,
                        help="Generate for a specific geometry only (overrides config)")
    parser.add_argument("--num-particles", type=int, default=None,
                        help="Generate for a specific N only (overrides config)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic Oseen approximation instead of libMobility")
    args = parser.parse_args()

    config = load_yaml(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    use_synthetic = args.synthetic or config.get("use_synthetic", False)

    geometries = [args.geometry] if args.geometry else config.get("geometries", [])
    particle_counts = [args.num_particles] if args.num_particles else config.get("num_particles", [64])

    if not geometries:
        print("Error: no geometries specified in config or --geometry flag.")
        sys.exit(1)

    for geo in geometries:
        for N in particle_counts:
            print(f"Generating {geo} with N={N} ({'synthetic' if use_synthetic else 'libMobility'})...")
            generate_dataset_for_geometry(geo, config, N=N, use_synthetic=use_synthetic)

    print("Done!")


if __name__ == "__main__":
    main()
