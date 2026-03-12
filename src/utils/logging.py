"""Lightweight logging wrapper supporting TensorBoard and Weights & Biases backends."""

from pathlib import Path
from typing import Optional

import torch


class Logger:
    """Unified logging wrapper for tensorboard and wandb."""

    def __init__(self, config, backend: str = "tensorboard", log_dir: str = "logs"):
        self.backend = backend
        self._writer = None
        self._run = None

        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=log_dir)
        elif backend == "wandb":
            import wandb
            self._run = wandb.init(
                project="hydro-ml",
                config=vars(config) if hasattr(config, "__dict__") else config,
            )

    def log_scalar(self, tag: str, value: float, step: int):
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)
        if self._run is not None:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_dict(self, metrics: dict, step: int):
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def finish(self):
        if self._writer is not None:
            self._writer.close()
        if self._run is not None:
            import wandb
            wandb.finish()
