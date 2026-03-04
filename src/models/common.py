"""Shared model building blocks: MLP with optional layer norm and residual."""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation, layer norm, dropout, and residual."""

    def __init__(
        self,
        dims: List[int],
        activation: type = nn.SiLU,
        layer_norm: bool = False,
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation/norm after final layer
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.residual and x.shape[-1] == out.shape[-1]:
            out = out + x
        return out
