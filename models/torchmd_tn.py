"""TensorNet model for learning the hydrodynamic mobility operator.

Wraps TorchMD-NET's TensorNet backbone to predict per-particle displacements
from forces and positions. The backbone processes only positions to produce
per-node features, then a MobilityHead predicts displacement = M(X) · F,
enforcing linearity in forces by construction.
"""

import torch
import torch.nn as nn
from torchmdnet.models.tensornet import TensorNet

from models.mobility_head import MobilityHead


class HydroTorchMD_TN(nn.Module):
    """TensorNet for predicting hydrodynamic displacements.

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

        self.backbone.tensor_embedding.emb=nn.Linear(3, cfg.hidden_dim)

        self.output_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 3)
        )

        mobility_hidden = getattr(cfg, "mobility_hidden_dim", 64)
        self.mobility_head = MobilityHead(
            node_dim=cfg.hidden_dim,
            hidden_dim=mobility_hidden,
            cutoff_upper=cfg.cutoff_radius,
        )

        # Hook to capture edge data from backbone's distance module
        self._edge_cache = {}

        def _capture_edges(module, input, output):
            self._edge_cache["edge_index"] = output[0]
            self._edge_cache["edge_weight"] = output[1]
            self._edge_cache["edge_vec"] = output[2]

        self.backbone.distance.register_forward_hook(_capture_edges)

    def forward(self, data):
        box = getattr(data, "box_vecs", None)
        if box is not None:
            box = box.view(-1, 3, 3)

        # Backbone: positions only -> node features
        N = data.pos.shape[0]
        q = torch.zeros(N, device=data.pos.device, dtype=data.pos.dtype)
        x, _, _, _, _ = self.backbone(
            z=data.forces, pos=data.pos, batch=data.batch, box=box, q=q
        )

        # MobilityHead: M(positions) · F
        return self.output_head(x)
