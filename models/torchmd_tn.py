"""TensorNet model for learning the hydrodynamic mobility operator.

Wraps TorchMD-NET's TensorNet backbone to predict per-particle displacements
from forces and positions.  Forces are injected into the backbone's tensor
features as antisymmetric (vector-channel) tensors after the embedding layer.
The interaction layers then propagate force information through the geometric
structure.  A MobilityHead applies forces linearly to produce displacements.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torchmdnet.models.tensornet import TensorNet, vector_to_skewtensor
from torchmdnet.models.utils import CosineCutoff, rbf_class_mapping


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

        # Geometry conditioning: embed geometry_id (0=nbody_open, 1=pse_periodic)
        # into a learned vector that is added to every node's features.
        # This lets the MobilityHead learn different α, β for each geometry.
        num_geometries = getattr(cfg, "num_geometries", 2)
        self.geometry_embedding = nn.Embedding(num_geometries, cfg.hidden_dim)

        # ---- inline mobility output with RBF + deeper MLP ----------------
        # RBF expansion gives the edge MLP direct access to pairwise distances,
        # so it can learn the 1/r and 1/r³ distance dependence of the RPY tensor.
        self.rbf = rbf_class_mapping[cfg.rbf_type](
            cutoff_lower=0.0,
            cutoff_upper=cfg.cutoff_radius,
            num_rbf=cfg.num_rbf,
            trainable=cfg.trainable_rbf,
        )

        # Edge MLP: [h_dst || h_src || RBF(r)] → 2 scalars (α, β)
        # 2 hidden layers for more expressive distance-dependent prediction.
        mobility_hidden = getattr(cfg, "mobility_hidden_dim", 64)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * cfg.hidden_dim + cfg.num_rbf, mobility_hidden),
            nn.SiLU(),
            nn.Linear(mobility_hidden, mobility_hidden),
            nn.SiLU(),
            nn.Linear(mobility_hidden, 2),
        )
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)

        self.cutoff = CosineCutoff(cutoff_lower=0.0, cutoff_upper=cfg.cutoff_radius)

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

        # Geometry conditioning: broadcast per-graph geometry_id to per-node,
        # embed, and add to node features so MobilityHead can specialise.
        geo_id = data.geometry_id.long().view(-1)            # [B]
        geo_per_node = geo_id[data.batch]                  # [N]
        x = x + self.geometry_embedding(geo_per_node)      # [N, H]

        # Retrieve edges captured by distance hook
        edge_index = self._edge_cache["edge_index"]
        edge_weight = self._edge_cache["edge_weight"]
        edge_vec = self._edge_cache["edge_vec"]

        # Mobility output: predict α, β per edge from node features + RBF distances
        src, dst = edge_index
        rbf = self.rbf(edge_weight)                                    # [E, num_rbf]
        h_ij = torch.cat([x[dst], x[src], rbf], dim=-1)               # [E, 2H + rbf]
        ab = self.edge_mlp(h_ij)                                       # [E, 2]
        alpha, beta = ab[:, 0], ab[:, 1]

        # Smooth distance cutoff
        cutoff_vals = self.cutoff(edge_weight)
        alpha = alpha * cutoff_vals
        beta = beta * cutoff_vals

        # Unit direction vectors; zero for self-loops
        safe_dist = edge_weight.clamp(min=1e-8).unsqueeze(-1)
        r_hat = edge_vec / safe_dist
        r_hat = r_hat * (src != dst).unsqueeze(-1).float()

        F_src = data.forces[src]

        # dR_i = Σ_j [α_ij · F_j + β_ij · (r̂_ij · F_j) · r̂_ij]
        iso = alpha.unsqueeze(-1) * F_src
        r_dot_F = (r_hat * F_src).sum(dim=-1, keepdim=True)
        aniso = beta.unsqueeze(-1) * r_dot_F * r_hat

        N = data.pos.size(0)
        return scatter(iso + aniso, dst, dim=0, dim_size=N, reduce="sum")
