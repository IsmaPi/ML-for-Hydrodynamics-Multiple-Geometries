"""From-scratch reimplementations of TorchMD-NET architectures.

Uses only low-level building blocks (OptimizedDistance, RBF, CosineCutoff,
scatter) from torchmdnet.models.utils plus standard PyTorch. All interaction
layers, continuous-filter convolutions, and equivariant attention are
implemented explicitly here for full transparency.
"""

from .torchmd_gn import HydroTorchMD_GN_FromZero
from .torchmd_et import HydroTorchMD_ET_FromZero

__all__ = ["HydroTorchMD_GN_FromZero", "HydroTorchMD_ET_FromZero"]
