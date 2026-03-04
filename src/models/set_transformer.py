"""Set Transformer for hydrodynamic displacement prediction.

Implements the architecture described in Section 1.5.2 of the methodology,
based on Lee et al. (2019) "Set Transformer: A Framework for Attention-based
Permutation-Invariant Neural Networks".

Key difference from standard Set Transformer: we need per-particle output
(equivariant), NOT a pooled set-level summary (invariant). Therefore we use
SAB/ISAB layers for inter-particle attention and apply a per-element MLP
readout instead of PMA pooling.

Following Equation (13):
  DeltaQ_hat = SetTransformer({x_i}_{i=1}^N) in R^{3N}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


# ===================================================================
# Core attention blocks (Lee et al., 2019)
# ===================================================================
class MAB(nn.Module):
    """Multihead Attention Block."""

    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, ln: bool = False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln0 = nn.LayerNorm(dim_V) if ln else None
        self.ln1 = nn.LayerNorm(dim_V) if ln else None

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        Q_proj = self.fc_q(Q)
        K_proj = self.fc_k(K)
        V_proj = self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q_proj.split(dim_split, 2), 0)
        K_ = torch.cat(K_proj.split(dim_split, 2), 0)
        V_ = torch.cat(V_proj.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        O = O if self.ln0 is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if self.ln1 is None else self.ln1(O)
        return O


class SAB(nn.Module):
    """Set Attention Block: self-attention over set elements. O(N^2)."""

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class ISAB(nn.Module):
    """Induced Set Attention Block: reduces complexity to O(Nm). (Lee et al., 2019)

    Uses m trainable inducing points to avoid quadratic attention:
      1. Inducing points attend to input set -> H
      2. Input set attends to transformed inducing points -> output
    """

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int, ln: bool = False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


# ===================================================================
# Full Set Transformer model
# ===================================================================
class HydroSetTransformer(nn.Module):
    """Set Transformer for per-particle displacement prediction.

    Architecture:
      1. Input encoder: Linear + LayerNorm + SiLU -> hidden_dim
      2. L attention layers (ISAB or SAB)
      3. Per-element readout: MLP -> 3D displacement per particle

    Note: No PMA pooling — we need per-particle output, not a set summary.
    The SAB/ISAB layers produce updated per-element embeddings that encode
    global context through attention, and the per-element MLP readout maps
    each particle's embedding to its predicted displacement.
    """

    def __init__(self, config):
        super().__init__()

        input_dim = config.input_dim
        hidden_dim = config.hidden_dim
        num_heads = config.num_heads
        num_layers = config.num_layers
        num_inds = getattr(config, "num_inducing_points", 32)
        use_isab = getattr(config, "use_isab", True)
        use_ln = config.layer_norm
        output_dim = config.output_dim

        # Input encoder
        encoder_layers = [nn.Linear(input_dim, hidden_dim)]
        if use_ln:
            encoder_layers.append(nn.LayerNorm(hidden_dim))
        encoder_layers.append(nn.SiLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_isab:
                self.layers.append(ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=use_ln))
            else:
                self.layers.append(SAB(hidden_dim, hidden_dim, num_heads, ln=use_ln))

        # Per-element readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Accepts PyG batched Data. Converts flat (total_nodes, D) to
        (B, N_max, D) for attention, then back to flat for output.

        Args:
            data: PyG Data with x (node features) and batch indices.
        Returns:
            Predicted displacements (total_nodes, output_dim).
        """
        x = data.x        # (total_nodes, D)
        batch = data.batch  # (total_nodes,)

        # Convert to padded batch tensor for attention
        x_batched, mask = self._to_batch_tensor(x, batch)

        # Encode
        h = self.encoder(x_batched)

        # Attention layers
        for layer in self.layers:
            h = layer(h)

        # Per-element readout
        pred = self.readout(h)

        # Convert back to flat
        return self._from_batch_tensor(pred, mask)

    def _to_batch_tensor(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> tuple:
        """Convert flat (total_nodes, D) + batch -> (B, N_max, D) + mask."""
        batch_size = batch.max().item() + 1
        counts = torch.bincount(batch, minlength=batch_size)
        N_max = counts.max().item()
        D = x.shape[-1]

        padded = torch.zeros(batch_size, N_max, D, device=x.device, dtype=x.dtype)
        mask = torch.zeros(batch_size, N_max, dtype=torch.bool, device=x.device)

        # Efficiently fill padded tensor
        cumsum = torch.cat([torch.tensor([0], device=x.device), counts.cumsum(0)])
        for i in range(batch_size):
            n_i = counts[i].item()
            padded[i, :n_i] = x[cumsum[i]:cumsum[i] + n_i]
            mask[i, :n_i] = True

        return padded, mask

    def _from_batch_tensor(
        self, padded: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Convert (B, N_max, D) + mask -> flat (total_nodes, D)."""
        outputs = []
        for i in range(padded.shape[0]):
            n_i = mask[i].sum().item()
            outputs.append(padded[i, :n_i])
        return torch.cat(outputs, dim=0)
