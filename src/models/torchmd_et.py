"""TorchMD-ET wrapper for hydrodynamic displacement prediction.

Replaces the hand-rolled Set Transformer with TorchMD-NET's Equivariant
Transformer. Key design choices:

  - Neighbor finding handled internally by OptimizedDistance (CUDA-native)
  - Equivariant vector output vec (N, 3, hidden_dim) projected to (N, 3)
    This is physically correct: displacements transform as 3-vectors under
    rotation, so using the equivariant branch gives consistent predictions
  - backbone.embedding replaced with nn.Linear for continuous 9-dim features
    [force(3), theta(6)]
  - PBC supported via box=(3,3) tensor passed at forward time
"""

import torch
import torch.nn as nn
from torchmdnet.models.torchmd_et import TorchMD_ET


class HydroTorchMD_ET(nn.Module):
    """Equivariant Transformer for per-particle displacement prediction.

    Input:
        data.x:       (N, 9) node features [force(3), theta(6)]
        data.pos:     (N, 3) positions (used for neighbor finding only)
        data.batch:   (N,) batch index
        data.box_vecs: (3,3) box matrix for PBC, or None

    Output:
        (N, 3) predicted displacement per particle (equivariant)
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone = TorchMD_ET(
            hidden_channels=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_rbf=cfg.num_rbf,
            rbf_type=cfg.rbf_type,
            trainable_rbf=cfg.trainable_rbf,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=False,
            num_heads=cfg.num_heads,
            distance_influence=cfg.distance_influence,
            cutoff_lower=0.0,
            cutoff_upper=cfg.cutoff_radius,
            max_z=2,                     # dummy — replaced below
            max_num_neighbors=cfg.max_num_neighbors,
        )

        # Replace integer atom embedding with continuous linear projection
        self.backbone.embedding = nn.Linear(cfg.input_dim, cfg.hidden_dim)

        # Equivariant projection: (N, 3, hidden_dim) -> (N, 3)
        # Applying Linear to last dim is equivariant: rotation of pos → same
        # rotation of vec → same weighted sum → correct displacement direction
        self.vec_proj = nn.Linear(cfg.hidden_dim, 1, bias=False)

    def forward(self, data):
        box = getattr(data, "box_vecs", None)
        # backbone returns (x, vec, z, pos, batch)
        # vec: (N, 3, hidden_dim) — equivariant vector features
        _, vec, _, _, _ = self.backbone(
            z=data.x, pos=data.pos, batch=data.batch, box=box
        )
        # Project hidden_dim -> 1 scalar per spatial direction: (N, 3, 1) -> (N, 3)
        return self.vec_proj(vec).squeeze(-1)
