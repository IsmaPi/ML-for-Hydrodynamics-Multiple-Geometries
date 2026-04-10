"""Shared mobility output head used by the ET and GN models.

Predicts two scalar coefficients (alpha, beta) per edge from concatenated
node features, then computes displacement as a sum of isotropic and
anisotropic terms. Note: the TensorNet model uses its own inline output
head with RBF features instead of this class.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torchmdnet.models.utils import CosineCutoff


class MobilityHead(nn.Module):

    def __init__(self, node_dim, hidden_dim=64, cutoff_upper=30.0):
        super().__init__()

        # MLP: [h_i || h_j] -> 2 scalars (alpha, beta)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)
        self.cutoff = CosineCutoff(cutoff_lower=0.0, cutoff_upper=cutoff_upper)

    def forward(self, node_features, edge_index, edge_weight, edge_vec, forces):
        src, dst = edge_index

        # Per-edge features: node pair concatenation (no RBF)
        h_ij = torch.cat([node_features[dst], node_features[src]], dim=-1)
        ab = self.edge_mlp(h_ij)
        alpha, beta = ab[:, 0], ab[:, 1]

        # Apply distance cutoff
        cutoff_vals = self.cutoff(edge_weight)
        alpha = alpha * cutoff_vals
        beta = beta * cutoff_vals

        # Unit direction vectors; zero for self-loops
        safe_dist = edge_weight.clamp(min=1e-8).unsqueeze(-1)
        r_hat = edge_vec / safe_dist
        r_hat = r_hat * (src != dst).unsqueeze(-1).float()

        F_src = forces[src]

        # Isotropic + anisotropic interaction terms
        iso = alpha.unsqueeze(-1) * F_src
        r_dot_F = (r_hat * F_src).sum(dim=-1, keepdim=True)
        aniso = beta.unsqueeze(-1) * r_dot_F * r_hat

        N = node_features.size(0)
        return scatter(iso + aniso, dst, dim=0, dim_size=N, reduce="sum")
