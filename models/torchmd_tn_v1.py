"""TensorNet model for learning the hydrodynamic mobility operator.

Wraps TorchMD-NET's TensorNet backbone to predict per-particle displacements
from forces and positions.  Forces are injected into the backbone's tensor
features as antisymmetric (vector-channel) tensors after the embedding layer.
The interaction layers then propagate force information through the geometric
structure.  A MobilityHead applies forces linearly to produce displacements.
"""

import torch
import torch.nn as nn
from torchmdnet.models.tensornet import TensorNet, vector_to_skewtensor

from models.mobility_head import MobilityHead


class HydroTorchMD_TN(nn.Module):
    """TensorNet for predicting hydrodynamic displacements.

    Forces enter the backbone as antisymmetric tensors (the vector channel of
    the I/A/S decomposition) injected after the tensor embedding.  Interaction
    layers propagate the combined force + geometry features.  MobilityHead
    produces displacement = M(X) · F.

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

        # Learnable per-channel weights for force injection into the
        # antisymmetric (A) tensor channel.  Zero-init keeps initial
        # behaviour identical to a position-only backbone.
        self.force_channels = nn.Parameter(torch.zeros(cfg.hidden_dim))

        mobility_hidden = getattr(cfg, "mobility_hidden_dim", 64)
        self.mobility_head = MobilityHead(
            node_dim=cfg.hidden_dim,
            hidden_dim=mobility_hidden,
            cutoff_upper=cfg.cutoff_radius,
        )

        # ---- hooks ----------------------------------------------------
        # Capture edge data from backbone's distance module
        self._edge_cache = {}

        def _capture_edges(module, input, output):
            self._edge_cache["edge_index"] = output[0]
            self._edge_cache["edge_weight"] = output[1]
            self._edge_cache["edge_vec"] = output[2]

        self.backbone.distance.register_forward_hook(_capture_edges)

        # Inject force-derived antisymmetric tensors after tensor embedding.
        # The hook adds  skew(F_i * w)  to the position-derived tensor X.
        self._cached_forces = None

        def _inject_force_tensor(module, input, output):
            # output: X  [N, 3, 3, H]  from tensor embedding
            # forces:     [N, 3]
            # force_channels: [H]
            f = self._cached_forces                       # [N, 3]
            fe = f.unsqueeze(-1) * self.force_channels    # [N, 3, H]
            return output + vector_to_skewtensor(fe)      # [N, 3, 3, H]

        self.backbone.tensor_embedding.register_forward_hook(
            _inject_force_tensor
        )

    def forward(self, data):
        box = getattr(data, "box_vecs", None)
        if box is not None:
            box = box.view(-1, 3, 3)

        # Store forces so the tensor-embedding hook can access them
        self._cached_forces = data.forces

        # Backbone: uniform atom types + positions  →  node features
        # (forces enter via the tensor-embedding hook above)
        x, _, _, _, _ = self.backbone(
            z=data.x, pos=data.pos, batch=data.batch, box=box
        )

        # Retrieve edges captured by distance hook
        edge_index = self._edge_cache["edge_index"]
        edge_weight = self._edge_cache["edge_weight"]
        edge_vec = self._edge_cache["edge_vec"]

        # MobilityHead: M(features) · F
        return self.mobility_head(
            x, edge_index, edge_weight, edge_vec, data.forces
        )
