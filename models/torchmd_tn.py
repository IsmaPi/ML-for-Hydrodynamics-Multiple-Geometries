"""TensorNet model for learning the hydrodynamic mobility operator.

Wraps TorchMD-NET's TensorNet backbone to predict per-particle displacements
from forces and positions. Uses equivariant scalar coefficients:

    displacement = self_mobility * F
                 + alpha * F               (self-mobility correction)
                 + beta * neighbor_dir     (interaction correction)

Unlike TorchMD_ET, TensorNet only outputs invariant scalar features (no vector
features), so the equivariant output relies entirely on scalar coefficients
multiplied by equivariant basis vectors.

The model learns: displacement = M(X) . F
where M is the configuration-dependent mobility tensor.
"""

import math

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torchmdnet.models.tensornet import TensorNet


class HydroTorchMD_TN(nn.Module):
    """TensorNet for predicting hydrodynamic displacements.

    Takes particle positions and forces, predicts displacement = M(X) . F.
    Output is equivariant: invariant scalars x equivariant vectors.

    Output = self_mobility * F
             + alpha * F                  (self-mobility correction)
             + beta * neighbor_dir        (interaction correction)

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

        # Equivariant scalar path: predict 2 scalar coefficients from invariant features
        # coeff 0: multiplies force vector F_i (self-mobility correction)
        # coeff 1: multiplies neighbor direction (interaction correction)
        self.scalar_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 2),
        )

        # Zero-init last layer so model starts from pure self-mobility baseline
        nn.init.zeros_(self.scalar_head[-1].weight)
        nn.init.zeros_(self.scalar_head[-1].bias)

    def _compute_neighbor_directions(self, pos, batch, box):
        """Compute mean neighbor direction for each particle.

        Uses the backbone's distance module to get the same neighbor graph,
        then aggregates normalized displacement vectors per particle.

        Returns:
            (N, 3) mean neighbor direction per particle (equivariant).
            Zero vector for isolated particles with no neighbors.
        """
        edge_index, edge_weight, edge_vec = self.backbone.distance(pos, batch, box)

        # Filter out self-loops
        non_self = edge_index[0] != edge_index[1]
        edge_idx = edge_index[:, non_self]
        edge_v = edge_vec[non_self]

        # Normalize to unit direction vectors r_hat_ij
        norms = edge_v.norm(dim=1, keepdim=True).clamp(min=1e-12)
        edge_v_unit = edge_v / norms

        # Aggregate: mean neighbor direction per receiver particle
        N = pos.shape[0]
        neighbor_dir = scatter(edge_v_unit, edge_idx[1], dim=0, dim_size=N, reduce="mean")

        return neighbor_dir

    def forward(self, data):
        box = getattr(data, "box_vecs", None)

        # Backbone returns (scalar_features, None, z, pos, batch)
        # TensorNet does not output vector features
        # Pass q=zeros(N) explicitly because TensorNet defaults to q=zeros_like(z),
        # which would be (N,3) instead of (N,) when z is our 2D force features
        N = data.x.shape[0]
        q = torch.zeros(N, device=data.x.device, dtype=data.x.dtype)
        x_scalar, _, _, _, _ = self.backbone(
            z=data.x, pos=data.pos, batch=data.batch, box=box, q=q
        )

        # Compute equivariant basis vectors
        neighbor_dir = self._compute_neighbor_directions(data.pos, data.batch, box)

        # Scalar path: 2 invariant coefficients x equivariant directions
        coeffs = self.scalar_head(x_scalar)  # (N, 2)
        scalar_out = (
            coeffs[:, 0:1] * data.x           # alpha * F (self-mobility correction)
            + coeffs[:, 1:2] * neighbor_dir    # beta * r_hat_agg (interaction correction)
        )

        # Delta learning: analytical self-mobility + equivariant corrections
        return self.self_mobility * data.x + scalar_out
