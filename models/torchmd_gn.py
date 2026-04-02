"""SchNet-style GNN model for learning the hydrodynamic mobility operator.

Wraps TorchMD-NET's TorchMD_GN backbone to predict per-particle displacements
from forces and positions. The backbone processes only positions to produce
per-node features, then a MobilityHead predicts displacement = M(X) · F,
enforcing linearity in forces by construction.
"""

import torch.nn as nn
from torchmdnet.models.torchmd_gn import TorchMD_GN
from torchmdnet.models.utils import OptimizedDistance

from models.mobility_head import MobilityHead


class HydroTorchMD_GN(nn.Module):
    """SchNet-style GNN for predicting hydrodynamic displacements.

    Forward input:
        data.pos:      (N, 3) particle positions
        data.x:        (N,) integer atom types (all ones)
        data.forces:   (N, 3) per-particle forces
        data.batch:    (N,) batch assignment
        data.box_vecs: (3, 3) periodic box or None

    Forward output:
        (N, 3) predicted displacement per particle
    """

    def __init__(self, cfg):
        super().__init__()

        cfg.hidden_dim = 3

        self.backbone = TorchMD_GN(
            hidden_channels=cfg.hidden_dim,
            num_filters=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_rbf=cfg.num_rbf,
            rbf_type=cfg.rbf_type,
            trainable_rbf=cfg.trainable_rbf,
            activation="silu",
            neighbor_embedding= False,
            cutoff_lower=0.0,
            cutoff_upper=cfg.cutoff_radius,
            max_z=2,
            max_num_neighbors=cfg.max_num_neighbors,
        )
        self.backbone.embedding = nn.Identity()
        # GN backbone doesn't return edge vectors, so we need a separate
        # distance module with return_vecs=True for the MobilityHead
        self.distance_vecs = OptimizedDistance(
            cutoff_lower=0.0,
            cutoff_upper=cfg.cutoff_radius,
            max_num_pairs=-cfg.max_num_neighbors,
            return_vecs=True,
            loop=True,
            long_edge_index=True,
        )

        mobility_hidden = getattr(cfg, "mobility_hidden_dim", 64)
        self.mobility_head = MobilityHead(
            node_dim=cfg.hidden_dim,
            hidden_dim=mobility_hidden,
            cutoff_upper=cfg.cutoff_radius,
        )

    def forward(self, data):
        box = getattr(data, "box_vecs", None)
        if box is not None:
            box = box.view(-1, 3, 3)

        # Backbone: positions only -> node features
        x, _, _, _, _ = self.backbone(
            z=data.forces, pos=data.pos, batch=data.batch, box=box
        )

        # MobilityHead: M(positions) · F
        return x
