"""Equivariant Transformer model for learning the hydrodynamic mobility operator.

Wraps TorchMD-NET's TorchMD_ET backbone to predict per-particle displacements
from forces and positions. The ET architecture uses equivariant vector features
that can represent anisotropic interactions like the RPY mobility tensor's
b(r) * r_hat ⊗ r_hat term.

The model learns: displacement = M(X) · F
where M is the configuration-dependent mobility tensor.
"""

import math

import torch
import torch.nn as nn
from torchmdnet.models.torchmd_et import TorchMD_ET


class HydroTorchMD_ET(nn.Module):
    """Equivariant Transformer for predicting hydrodynamic displacements.

    Takes particle positions and forces, predicts displacement = M(X) · F.
    Uses equivariant vector features to capture direction-dependent mobility.

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

        # Analytical self-mobility: U_self = F / (6πηa)
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

        # Equivariant scalar path: scalar correction magnitude along force direction
        # Outputs (N, 1) scalar that multiplies normalized force vector
        self.scalar_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

        # Equivariant vector path: anisotropic interaction correction
        # (N, 3, hidden_dim) -> (N, 3, 1) -> (N, 3)
        self.output_proj = nn.Linear(cfg.hidden_dim, 1, bias=False)

        # Zero-init both paths so model starts from pure self-mobility baseline
        nn.init.zeros_(self.scalar_head[-1].weight)
        nn.init.zeros_(self.scalar_head[-1].bias)
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, data):
        box = getattr(data, "box_vecs", None)

        # Backbone returns (scalar_features, vector_features, z, pos, batch)
        x_scalar, vec, _, _, _ = self.backbone(
            z=data.x, pos=data.pos, batch=data.batch, box=box
        )

        # Equivariant scalar path: correction along force direction
        # For zero-force particles, scalar_out = 0 (no force direction)
        scalar_mag = self.scalar_head(x_scalar)  # (N, 1)
        force_norm = data.x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        force_dir = data.x / force_norm
        # Zero out for particles with negligible force
        mask = (force_norm > 1e-6).float()
        scalar_out = scalar_mag * force_dir * mask  # (N, 3), equivariant

        # Equivariant vector path: anisotropic interaction correction
        vector_out = self.output_proj(vec).squeeze(-1)  # (N, 3)

        # Delta learning: self-mobility + equivariant scalar + equivariant vector
        return self.self_mobility * data.x + scalar_out + vector_out
