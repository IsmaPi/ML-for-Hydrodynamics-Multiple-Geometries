"""TorchMD-GN wrapper for hydrodynamic displacement prediction.

Replaces the hand-rolled GNN with a production-quality SchNet-style
graph network from TorchMD-NET. Key design choices:

  - Neighbor finding handled internally by OptimizedDistance (CUDA-native)
  - No kNN or radius graph construction in user code
  - RBF-encoded pairwise distances + directional vectors as edge features
  - backbone.embedding replaced with nn.Linear so continuous 9-dim features
    [force(3), theta(6)] pass through without integer lookup
  - PBC supported: pass box=(3,3) tensor for periodic geometries, None for open
"""

import torch
import torch.nn as nn
from torchmdnet.models.torchmd_gn import TorchMD_GN


class HydroTorchMD_GN(nn.Module):
    """SchNet-style GNN for per-particle displacement prediction.

    Input:
        data.x:       (N, 9) node features [force(3), theta(6)]
        data.pos:     (N, 3) positions (used for neighbor finding only)
        data.batch:   (N,) batch index
        data.box_vecs: (3,3) box matrix for PBC, or None

    Output:
        (N, 3) predicted displacement per particle
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone = TorchMD_GN(
            hidden_channels=cfg.hidden_dim,
            num_filters=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_rbf=cfg.num_rbf,
            rbf_type=cfg.rbf_type,
            trainable_rbf=cfg.trainable_rbf,
            activation="silu",
            neighbor_embedding=False,
            cutoff_lower=0.0,
            cutoff_upper=cfg.cutoff_radius,
            max_z=2,                     # dummy — replaced below
            max_num_neighbors=cfg.max_num_neighbors,
        )

        # Replace integer atom embedding with continuous linear projection
        self.backbone.embedding = nn.Linear(cfg.input_dim, cfg.hidden_dim)

        # Output head: scalar node features -> per-particle displacement
        self.output_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(self, data):
        box = getattr(data, "box_vecs", None)
        # backbone returns (x, v, z, pos, batch) — only x (scalar) used here
        x, _, _, _, _ = self.backbone(
            z=data.x, pos=data.pos, batch=data.batch, box=box
        )
        return self.output_head(x)   # (N, 3)
