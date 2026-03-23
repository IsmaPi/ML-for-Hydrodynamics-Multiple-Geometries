"""Equivariant Transformer model for learning the hydrodynamic mobility operator.

Wraps TorchMD-NET's TorchMD_ET backbone to predict per-particle displacements
from forces and positions. Uses scalar message passing with equivariant attention
to learn interaction corrections on top of analytical self-mobility.

The model learns: displacement = M(X) · F
where M is the configuration-dependent mobility tensor.
"""

import math

import torch.nn as nn
from torchmdnet.models.torchmd_et import TorchMD_ET


class HydroTorchMD_ET(nn.Module):
    """Equivariant Transformer for predicting hydrodynamic displacements.

    Takes particle positions and forces, predicts displacement = M(X) · F.
    Uses scalar features from ET backbone to predict 3D displacement correction.

    Output = self_mobility * F + output_head(scalar_features)

    Args:
        cfg: ModelConfig with architecture hyperparameters.

    Forward input:
        data.x:        (N, input_dim) per-particle forces
        data.pos:      (N, 3) particle positions
        data.batch:    (N,) batch assignment
        data.box_vecs: (3, 3) periodic box or None

    Forward output:
        (N, 3) predicted displacement per particle
    """

    def __init__(self, cfg):
        super().__init__()

        # Analytical self-mobility: U_self = F / (6*pi*eta*a)
        viscosity = getattr(cfg, "viscosity", 1.0)
        a = getattr(cfg, "hydrodynamic_radius", 1.0)
        self.self_mobility = 1.0 / (6.0 * math.pi * viscosity * a)

        # Input projection: continuous force features -> hidden dim
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)

        # TorchMD-NET Equivariant Transformer backbone
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
            max_z=2,
            max_num_neighbors=cfg.max_num_neighbors,
        )

        # Replace the default integer atom-type embedding with our input projection
        self.backbone.embedding = self.input_proj

        # Output head: scalar node features -> 3D displacement correction
        self.output_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

        # Zero-init last layer so model starts from pure self-mobility baseline
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

        # Equivariant vector path: project vector features to 3D correction
        # No bias to maintain equivariance; zero-init to start from working baseline
        self.vector_proj = nn.Linear(cfg.hidden_dim, 1, bias=False)
        nn.init.zeros_(self.vector_proj.weight)

    def forward(self, data):
        box = getattr(data, "box_vecs", None)

        # Backbone returns (scalar_features, vector_features, z, pos, batch)
        x, vec, _, _, _ = self.backbone(
            z=data.x, pos=data.pos, batch=data.batch, box=box
        )

        # vec: [N, 3, hidden_dim] → [N, 3, 1] → [N, 3]
        vector_correction = self.vector_proj(vec).squeeze(-1)

        # Delta learning: analytical self-mobility + learned corrections
        forces = data.x[:, :3]  # first 3 columns are forces (rest is geometry_id)
        return self.self_mobility * forces + self.output_head(x) + vector_correction
