"""SchNet-style GNN model for learning the hydrodynamic mobility operator.

Wraps TorchMD-NET's TorchMD_GN backbone to predict per-particle displacements
from forces and positions. SchNet uses isotropic (scalar) messages, so it can
only learn the isotropic part a(r)*I of the RPY tensor, not the anisotropic
b(r)*r_hat ⊗ r_hat term. Used as a baseline to compare against the ET.

The model learns: displacement = M(X) · F
where M is the configuration-dependent mobility tensor.
"""

import math

import torch
import torch.nn as nn
from torchmdnet.models.torchmd_gn import TorchMD_GN


class HydroTorchMD_GN(nn.Module):
    """SchNet-style GNN for predicting hydrodynamic displacements.

    Takes particle positions and forces, predicts displacement = M(X) · F.
    Uses scalar message passing — cannot capture anisotropic interactions.

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

        # TorchMD-NET SchNet backbone
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

    def forward(self, data):
        box = getattr(data, "box_vecs", None)

        # Backbone returns (scalar_features, vector_features, z, pos, batch)
        x, _, _, _, _ = self.backbone(
            z=data.x, pos=data.pos, batch=data.batch, box=box
        )

        # Delta learning: analytical self-mobility + learned interaction correction
        return self.self_mobility * data.x + self.output_head(x)
