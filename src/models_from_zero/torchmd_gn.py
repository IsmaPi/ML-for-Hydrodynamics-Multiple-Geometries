"""From-scratch reimplementation of TorchMD_GN (SchNet-style graph network).

Only low-level building blocks are imported from torchmdnet:
  - OptimizedDistance  (CUDA neighbor finding)
  - rbf_class_mapping  (GaussianSmearing / ExpNormalSmearing)
  - CosineCutoff       (smooth distance cutoff)
  - scatter            (graph aggregation)

Everything else — continuous-filter convolution, interaction blocks, the
full forward pass — is implemented explicitly here.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from torchmdnet.models.utils import (
    OptimizedDistance,
    rbf_class_mapping,
    CosineCutoff,
    scatter,
)


# ---------------------------------------------------------------------------
# Continuous-Filter Convolution (CFConv)
# ---------------------------------------------------------------------------
class CFConv(nn.Module):
    """SchNet continuous-filter convolution.

    For each edge (i, j):
        1. Compute filter weight W_ij = MLP(RBF(r_ij)) * cutoff(r_ij)
        2. Compute message m_ij = W_ij * Lin1(x_j)
        3. Aggregate: x_i' = sum_j m_ij
        4. Output: Lin2(x_i')
    """

    def __init__(self, in_channels, out_channels, num_filters, num_rbf,
                 cutoff_lower, cutoff_upper, activation_cls):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        # MLP that maps RBF features -> filter weights
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, num_filters),
            activation_cls(),
            nn.Linear(num_filters, num_filters),
        )
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

    def forward(self, x, edge_index, edge_weight, edge_attr, num_nodes):
        """
        Args:
            x:           (N, in_channels) node features
            edge_index:  (2, E) sender [0] / receiver [1] indices
            edge_weight: (E,) distances
            edge_attr:   (E, num_rbf) RBF-expanded distances
            num_nodes:   total number of nodes N

        Returns:
            (N, out_channels) updated node features
        """
        # Filter weights modulated by smooth cutoff
        C = self.cutoff(edge_weight)                     # (E,)
        W = self.filter_net(edge_attr) * C.unsqueeze(1)  # (E, num_filters)

        # Project node features and compute messages
        x_j = self.lin1(x).index_select(0, edge_index[1])  # (E, num_filters)
        msg = W * x_j                                       # (E, num_filters)

        # Aggregate messages to receiver nodes (edge_index[0])
        agg = scatter(msg, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum")

        return self.lin2(agg)  # (N, out_channels)


# ---------------------------------------------------------------------------
# Interaction Block
# ---------------------------------------------------------------------------
class InteractionBlock(nn.Module):
    """Single SchNet interaction layer: CFConv -> activation -> Linear."""

    def __init__(self, hidden_channels, num_filters, num_rbf,
                 cutoff_lower, cutoff_upper, activation_cls):
        super().__init__()
        self.conv = CFConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_filters=num_filters,
            num_rbf=num_rbf,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            activation_cls=activation_cls,
        )
        self.act = activation_cls()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, edge_attr, num_nodes):
        h = self.conv(x, edge_index, edge_weight, edge_attr, num_nodes)
        h = self.act(h)
        h = self.lin(h)
        return h  # caller adds residual: x = x + h


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
_activations = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}


class HydroTorchMD_GN_FromZero(nn.Module):
    """SchNet-style GNN for per-particle displacement prediction.

    Reimplemented from scratch. The only torchmdnet imports are the
    CUDA-native OptimizedDistance, RBF expansions, CosineCutoff, and scatter.

    Input:
        data.x:        (N, input_dim) node features [force(3), theta(6)]
        data.pos:      (N, 3) positions for neighbor finding
        data.batch:    (N,) batch index
        data.box_vecs: (3,3) PBC box or None

    Output:
        (N, 3) predicted displacement
    """

    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.hidden_dim
        act_cls = _activations.get(getattr(cfg, "activation", "silu"), nn.SiLU)

        # --- Neighbor finding (CUDA-native) ---
        self.distance = OptimizedDistance(
            cutoff_lower=0.0,
            cutoff_upper=cfg.cutoff_radius,
            max_num_pairs=-cfg.max_num_neighbors,
            return_vecs=False,
            loop=False,
            strategy=getattr(cfg, "neighbor_strategy", "brute"),
            include_transpose=True,
            resize_to_fit=True,
        )

        # --- RBF distance expansion ---
        RBF = rbf_class_mapping.get(cfg.rbf_type, rbf_class_mapping["expnorm"])
        self.distance_expansion = RBF(
            cutoff_lower=0.0,
            cutoff_upper=cfg.cutoff_radius,
            num_rbf=cfg.num_rbf,
            trainable=cfg.trainable_rbf,
        )

        # --- Input embedding (continuous features, not integer atom types) ---
        self.embedding = nn.Linear(cfg.input_dim, hidden)

        # --- Interaction layers ---
        self.interactions = nn.ModuleList([
            InteractionBlock(
                hidden_channels=hidden,
                num_filters=hidden,
                num_rbf=cfg.num_rbf,
                cutoff_lower=0.0,
                cutoff_upper=cfg.cutoff_radius,
                activation_cls=act_cls,
            )
            for _ in range(cfg.num_layers)
        ])

        # --- Output head ---
        self.output_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            act_cls(),
            nn.Linear(hidden, cfg.output_dim),
        )

    def forward(self, data):
        box = getattr(data, "box_vecs", None)

        # 1. Embed continuous node features
        x = self.embedding(data.x)  # (N, hidden)

        # 2. Compute neighbor list and distances
        edge_index, edge_weight, _ = self.distance(data.pos, data.batch, box)
        # edge_index: (2, E), edge_weight: (E,)

        # 3. Expand distances into RBF features
        edge_attr = self.distance_expansion(edge_weight)  # (E, num_rbf)

        # 4. Message-passing with residual connections
        num_nodes = data.x.size(0)
        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_weight, edge_attr, num_nodes)

        # 5. Per-particle displacement prediction
        return self.output_head(x)  # (N, 3)
