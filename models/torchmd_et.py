"""Equivariant Transformer model for hydrodynamics (experimental).

Uses a shared MobilityHead for the output. This architecture was explored
during development but not used for the final results. The TensorNet model
(torchmd_tn.py) was chosen instead for its tensor-Oseen correspondence.
"""

import torch.nn as nn
from torchmdnet.models.torchmd_et import TorchMD_ET

from models.mobility_head import MobilityHead


class HydroTorchMD_ET(nn.Module):
    """Equivariant Transformer for predicting hydrodynamic displacements.

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

        self.backbone = TorchMD_ET(
            hidden_channels=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_rbf=cfg.num_rbf,
            rbf_type=cfg.rbf_type,
            trainable_rbf=cfg.trainable_rbf,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=True,
            num_heads=cfg.num_heads,
            distance_influence=cfg.distance_influence,
            cutoff_lower=0.0,
            cutoff_upper=cfg.cutoff_radius,
            max_z=2,
            max_num_neighbors=cfg.max_num_neighbors,
        )

        mobility_hidden = getattr(cfg, "mobility_hidden_dim", 64)
        self.mobility_head = MobilityHead(
            node_dim=cfg.hidden_dim,
            hidden_dim=mobility_hidden,
            cutoff_upper=cfg.cutoff_radius
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
            # PyG concatenates per-graph (3,3) into (B*3, 3); reshape to (B, 3, 3)
            box = box.view(-1, 3, 3)

        # Backbone: positions only -> node features
        # z=data.x are integer atom types (all 1s), not forces
        x, vec, _, _, _ = self.backbone(
            z=data.x, pos=data.pos, batch=data.batch, box=box
        )

        # Get edges captured by hook
        edge_index = self._edge_cache["edge_index"]
        edge_weight = self._edge_cache["edge_weight"]
        edge_vec = self._edge_cache["edge_vec"]

        # MobilityHead: M(positions) · F
        return self.mobility_head(x, edge_index, edge_weight, edge_vec, data.forces)
