"""Message-passing Graph Neural Network for hydrodynamic displacement prediction.

Implements the architecture described in Section 1.5.1 of the methodology:
  - Node encoder: raw features -> hidden embeddings
  - Edge encoder: relative displacement + distance -> hidden edge embeddings
  - L message-passing layers with learned message (psi) and update (phi) functions
  - Per-particle readout head -> 3D translational displacement

Following Equations (10)-(12):
  x_i = [q_i, F_i, U_i^{n-1}, theta]
  m_ij^(l) = psi^(l)(h_i^(l), h_j^(l), e_ij)
  h_i^(l+1) = phi^(l)(h_i^(l), sum_j m_ij^(l))
  DeltaX_i = rho(h_i^(L))
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

from .common import MLP


class HydroMessagePassingLayer(MessagePassing):
    """Single message-passing layer.

    Message function psi: takes [h_i, h_j, e_ij] -> message vector
    Update function phi: takes [h_i, aggregated_messages] -> h_i'
    """

    def __init__(self, hidden_dim: int, edge_dim: int, layer_norm: bool = True, dropout: float = 0.0):
        super().__init__(aggr="add")

        self.message_mlp = MLP(
            [hidden_dim * 2 + edge_dim, hidden_dim, hidden_dim],
            layer_norm=layer_norm,
            dropout=dropout,
        )
        self.update_mlp = MLP(
            [hidden_dim * 2, hidden_dim, hidden_dim],
            layer_norm=layer_norm,
            dropout=dropout,
            residual=True,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """m_ij = psi(h_i, h_j, e_ij)."""
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """h_i' = phi(h_i, sum_j m_ij)."""
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class HydroGNN(nn.Module):
    """Full GNN model for hydrodynamic displacement prediction.

    Architecture:
      1. Node encoder: MLP [input_dim -> hidden_dim]
      2. Edge encoder: MLP [4 -> hidden_dim]
      3. L message-passing layers
      4. Readout head: MLP [hidden_dim -> output_dim] per particle
    """

    def __init__(self, config):
        super().__init__()

        input_dim = config.input_dim
        hidden_dim = config.hidden_dim
        edge_input_dim = 4  # relative_pos (3) + distance (1)
        num_layers = config.num_layers
        output_dim = config.output_dim
        layer_norm = config.layer_norm
        dropout = getattr(config, "dropout", 0.0)

        self.node_encoder = MLP(
            [input_dim, hidden_dim, hidden_dim],
            layer_norm=layer_norm,
        )

        self.edge_encoder = MLP(
            [edge_input_dim, hidden_dim, hidden_dim],
            layer_norm=layer_norm,
        )

        self.mp_layers = nn.ModuleList([
            HydroMessagePassingLayer(hidden_dim, hidden_dim, layer_norm, dropout)
            for _ in range(num_layers)
        ])

        self.readout = MLP(
            [hidden_dim, hidden_dim, hidden_dim // 2, output_dim],
            layer_norm=False,  # no normalization on final output
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyG Data with x (node features), edge_index, edge_attr.
        Returns:
            Predicted displacements (total_nodes, output_dim).
        """
        h = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        for mp_layer in self.mp_layers:
            h = mp_layer(h, data.edge_index, edge_attr)

        return self.readout(h)
