#!/usr/bin/env python3
"""CLI for generating training data using libMobility.

Usage:
    python scripts/generate_data.py --config configs/data/default.yaml
    python scripts/generate_data.py --config default
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_yaml, build_data_config, resolve_config_path
from data.generate import generate_all


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate hydrodynamic training data")
    parser.add_argument("--config", type=str, required=True,
                        help="Data config file or short name (e.g. 'default')")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config, "data")
    raw = load_yaml(config_path)
    data_cfg = build_data_config(raw)

    logging.info("Generating data for geometries: %s", data_cfg.geometries)
    logging.info("Particle counts: %s", data_cfg.num_particles)
    logging.info("Cloud samples per geo/N: %d, Pair samples per geo: %d",
                 data_cfg.num_cloud_samples, data_cfg.num_pair_samples)

    generate_all(data_cfg)

    logging.info("Data generation complete. Output: %s", data_cfg.output_dir)


if __name__ == "__main__":
    main()
