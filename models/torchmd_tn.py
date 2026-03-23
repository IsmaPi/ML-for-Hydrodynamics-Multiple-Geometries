"""TensorNet model for learning the hydrodynamic mobility operator.

Wraps TorchMD-NET's TensorNet backbone to predict per-particle displacements
from forces and positions. Uses scalar message passing with tensor product
interactions to learn interaction corrections on top of analytical self-mobility.

The model learns: displacement = M(X) · F
where M is the configuration-dependent mobility tensor.
"""

import math

import torch
import torch.nn as nn
from torchmdnet.models.tensornet import TensorNet


class HydroTorchMD_TN(nn.Module):
    """TensorNet for predicting hydrodynamic displacements.

    Takes particle positions and forces, predicts displacement = M(X) · F.
    Uses scalar features from TensorNet backbone to predict 3D displacement correction.

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

        # TorchMD-NET TensorNet backbone
        # static_shapes=False required because we pass 2D continuous features
        # instead of 1D integer atomic numbers
        self.backbone = TensorNet(
            hidden_channels=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_rbf=cfg.num_rbf,
            rbf_type=cfg.rbf_type,
            trainable_rbf=cfg.trainable_rbf,
            activation="silu",
            cutoff_lower=0.0,
            cutoff_upper=cfg.cutoff_radius,
            max_z=2,
            max_num_neighbors=cfg.max_num_neighbors,
            equivariance_invariance_group=getattr(cfg, "equivariance_invariance_group", "O(3)"),
            static_shapes=False,
        )

        # Replace the default integer atom-type embedding with our input projection
        self.backbone.tensor_embedding.emb = self.input_proj

        # Output head: scalar node features -> 3D displacement correction
        self.output_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

        # Zero-init last layer so model starts from pure self-mobility baseline
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

        # Equivariant vector path: extract vectors from antisymmetric tensor component
        # No bias to maintain equivariance; zero-init to start from working baseline
        self.vector_proj = nn.Linear(cfg.hidden_dim, 1, bias=False)
        nn.init.zeros_(self.vector_proj.weight)

        # Hook to capture final tensor X from last interaction layer
        self._last_X = None

        def _capture_X(module, input, output):
            self._last_X = output

        self.backbone.layers[-1].register_forward_hook(_capture_X)

    def forward(self, data):
        box = getattr(data, "box_vecs", None)

        # Backbone returns (scalar_features, None, z, pos, batch)
        # Pass q=zeros(N) explicitly because TensorNet defaults to q=zeros_like(z),
        # which would be (N,3) instead of (N,) when z is our 2D force features
        N = data.x.shape[0]
        q = torch.zeros(N, device=data.x.device, dtype=data.x.dtype)
        x, _, _, _, _ = self.backbone(
            z=data.x, pos=data.pos, batch=data.batch, box=box, q=q
        )

        # Extract equivariant vectors from captured tensor
        # Use raw antisymmetric part directly (avoids opt/non-opt decompose_tensor mismatch)
        X = self._last_X                                    # [N, 3, 3, hidden_channels]
        A = 0.5 * (X - X.transpose(1, 2))                   # antisymmetric part
        vec = torch.stack([A[:, 1, 2, :] - A[:, 2, 1, :],   # extract vector from skew-symmetric
                           A[:, 2, 0, :] - A[:, 0, 2, :],
                           A[:, 0, 1, :] - A[:, 1, 0, :]], dim=1)  # [N, 3, hidden_channels]
        vector_correction = self.vector_proj(vec).squeeze(-1)  # [N, 3]

        # Delta learning: analytical self-mobility + learned corrections
        forces = data.x[:, :3]  # first 3 columns are forces (rest is geometry_id)
        return self.self_mobility * forces + self.output_head(x) + vector_correction
