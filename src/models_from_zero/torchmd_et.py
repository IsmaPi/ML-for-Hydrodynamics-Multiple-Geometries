"""From-scratch reimplementation of TorchMD_ET (Equivariant Transformer).

Only low-level building blocks are imported from torchmdnet:
  - OptimizedDistance  (CUDA neighbor finding)
  - rbf_class_mapping  (GaussianSmearing / ExpNormalSmearing)
  - CosineCutoff       (smooth distance cutoff)
  - scatter            (graph aggregation)

Everything else — equivariant multi-head attention, vector feature updates,
the full forward pass — is implemented explicitly here.
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


_activations = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}


# ---------------------------------------------------------------------------
# Equivariant Multi-Head Attention
# ---------------------------------------------------------------------------
class EquivariantMultiHeadAttention(nn.Module):
    """Distance-dependent equivariant attention layer.

    Maintains both scalar features x (N, C) and equivariant vector
    features vec (N, 3, C). The vector channel transforms as a 3-vector
    under rotation — new directional information enters through the
    normalized edge vectors d_ij.

    Attention scores:
        attn_ij = act( sum_h q_i^h * k_j^h [* dk_ij^h] ) * cutoff(r_ij)

    Scalar update:
        dx_i = sum_j attn_ij * v_x_j

    Vector update:
        dvec_i = sum_j [ vec1_j * vec_j  +  vec2_j * d_ij ]
    """

    def __init__(self, hidden_channels, num_rbf, distance_influence,
                 num_heads, activation_cls, attn_activation_cls,
                 cutoff_lower, cutoff_upper):
        super().__init__()
        assert hidden_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.hidden_channels = hidden_channels
        self.distance_influence = distance_influence

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.act = activation_cls()
        self.attn_activation = attn_activation_cls()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        # Scalar projections
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        # v_proj outputs 3x hidden: split into (x_val, vec1_val, vec2_val)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        # Vector projections: 3 independent linear maps on vec's last dim
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        # Distance-dependent key/value modulation
        self.dk_proj = None
        if distance_influence in ("keys", "both"):
            self.dk_proj = nn.Linear(num_rbf, hidden_channels)
        self.dv_proj = None
        if distance_influence in ("values", "both"):
            self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3)

        # Output projection: 3x hidden -> split into (o1, o2, o3)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        """
        Args:
            x:          (N, C) scalar features
            vec:        (N, 3, C) equivariant vector features
            edge_index: (2, E) [sender, receiver]
            r_ij:       (E,) pairwise distances
            f_ij:       (E, num_rbf) RBF-expanded distances
            d_ij:       (E, 3) normalized direction vectors

        Returns:
            dx:    (N, C) scalar residual
            dvec:  (N, 3, C) vector residual
        """
        N = x.size(0)
        H = self.num_heads
        D = self.head_dim
        C = self.hidden_channels

        x_norm = self.layernorm(x)

        # --- Query, Key, Value (scalar path) ---
        q = self.q_proj(x_norm).view(N, H, D)           # (N, H, D)
        k = self.k_proj(x_norm).view(N, H, D)           # (N, H, D)
        v = self.v_proj(x_norm).view(N, H, D * 3)       # (N, H, 3D)

        # --- Vector feature projections ---
        # Apply 3 independent linear maps: vec (N, 3, C) -> (N, 3, 3C)
        vec_out = self.vec_proj(vec)                      # (N, 3, 3C)
        vec1, vec2, vec3 = vec_out.split(C, dim=-1)       # each (N, 3, C)

        # Dot product of vec1 and vec2 (contracted over spatial dim)
        vec_dot = (vec1 * vec2).sum(dim=1)                # (N, C)

        # --- Distance-dependent modulations ---
        dk = None
        if self.dk_proj is not None:
            dk = self.act(self.dk_proj(f_ij)).view(-1, H, D)   # (E, H, D)
        dv = None
        if self.dv_proj is not None:
            dv = self.act(self.dv_proj(f_ij)).view(-1, H, D * 3)  # (E, H, 3D)

        # --- Per-edge message computation ---
        src, dst = edge_index[0], edge_index[1]  # sender=dst in PyG convention
        # In TorchMD-NET: edge_index[0] = i (receiver), edge_index[1] = j (sender)
        q_i = q.index_select(0, src)              # (E, H, D)
        k_j = k.index_select(0, dst)              # (E, H, D)
        v_j = v.index_select(0, dst)              # (E, H, 3D)
        vec_j = vec.index_select(0, dst)           # (E, 3, C)
        vec_j = vec_j.view(-1, 3, H, D)           # (E, 3, H, D)

        # Attention scores
        if dk is not None:
            attn = (q_i * k_j * dk).sum(dim=-1)   # (E, H)
        else:
            attn = (q_i * k_j).sum(dim=-1)        # (E, H)
        attn = self.attn_activation(attn)

        # Apply distance cutoff
        cutoff_val = self.cutoff(r_ij).unsqueeze(1)  # (E, 1)
        attn = attn * cutoff_val                      # (E, H)

        # Modulate values by distance
        if dv is not None:
            v_j = v_j * dv                            # (E, H, 3D)

        # Split value into scalar and two vector components
        v_x, v_vec1, v_vec2 = v_j.split(D, dim=-1)
        # v_x:    (E, H, D) — scalar message
        # v_vec1: (E, H, D) — scales existing vec_j
        # v_vec2: (E, H, D) — creates new direction from d_ij

        # Scalar messages weighted by attention
        msg_x = v_x * attn.unsqueeze(2)               # (E, H, D)

        # Vector messages: preserve existing + add directional component
        # d_ij: (E, 3) -> (E, 3, 1, 1) for broadcasting
        d_ij_expanded = d_ij.unsqueeze(2).unsqueeze(3)  # (E, 3, 1, 1)
        msg_vec = (
            vec_j * v_vec1.unsqueeze(1)                 # (E, 3, H, D) scale existing
            + v_vec2.unsqueeze(1) * d_ij_expanded       # (E, 3, H, D) new direction
        )

        # --- Aggregate to receiver nodes ---
        agg_x = scatter(msg_x, src, dim=0, dim_size=N, reduce="sum")
        # agg_x: (N, H, D)
        agg_vec = scatter(msg_vec, src, dim=0, dim_size=N, reduce="sum")
        # agg_vec: (N, 3, H, D)

        # Reshape back to (N, C)
        agg_x = agg_x.reshape(N, C)
        agg_vec = agg_vec.reshape(N, 3, C)

        # --- Output projection ---
        o1, o2, o3 = self.o_proj(agg_x).split(C, dim=1)
        # o1, o2, o3: each (N, C)

        # Scalar residual: vec dot product gates o2, plus bias o3
        dx = vec_dot * o2 + o3                         # (N, C)

        # Vector residual: scale vec3 by o1, plus aggregated vector messages
        dvec = vec3 * o1.unsqueeze(1) + agg_vec        # (N, 3, C)

        return dx, dvec


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class HydroTorchMD_ET_FromZero(nn.Module):
    """Equivariant Transformer for per-particle displacement prediction.

    Reimplemented from scratch. The only torchmdnet imports are the
    CUDA-native OptimizedDistance, RBF expansions, CosineCutoff, and scatter.

    Maintains equivariant vector features vec (N, 3, C) alongside scalar
    features x (N, C). The final displacement is projected from vec —
    physically correct since displacements transform as 3-vectors.

    Input:
        data.x:        (N, input_dim) node features [force(3), theta(6)]
        data.pos:      (N, 3) positions for neighbor finding
        data.batch:    (N,) batch index
        data.box_vecs: (3,3) PBC box or None

    Output:
        (N, 3) predicted displacement (equivariant)
    """

    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.hidden_dim
        act_cls = _activations.get(getattr(cfg, "activation", "silu"), nn.SiLU)
        attn_act_cls = _activations.get(getattr(cfg, "attn_activation", "silu"), nn.SiLU)

        # --- Neighbor finding (CUDA-native) ---
        self.distance = OptimizedDistance(
            cutoff_lower=0.0,
            cutoff_upper=cfg.cutoff_radius,
            max_num_pairs=-cfg.max_num_neighbors,
            return_vecs=True,  # need directional vectors for equivariance
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

        # --- Input embedding (continuous features) ---
        self.embedding = nn.Linear(cfg.input_dim, hidden)

        # --- Equivariant attention layers ---
        self.attention_layers = nn.ModuleList([
            EquivariantMultiHeadAttention(
                hidden_channels=hidden,
                num_rbf=cfg.num_rbf,
                distance_influence=cfg.distance_influence,
                num_heads=cfg.num_heads,
                activation_cls=act_cls,
                attn_activation_cls=attn_act_cls,
                cutoff_lower=0.0,
                cutoff_upper=cfg.cutoff_radius,
            )
            for _ in range(cfg.num_layers)
        ])

        # --- Output normalization ---
        self.out_norm = nn.LayerNorm(hidden)

        # --- Equivariant output: project vec (N, 3, C) -> (N, 3) ---
        self.vec_proj = nn.Linear(hidden, 1, bias=False)

    def forward(self, data):
        box = getattr(data, "box_vecs", None)

        # 1. Embed continuous node features
        x = self.embedding(data.x)  # (N, hidden)

        # 2. Compute neighbor list, distances, AND direction vectors
        edge_index, edge_weight, edge_vec = self.distance(data.pos, data.batch, box)
        # edge_index: (2, E), edge_weight: (E,), edge_vec: (E, 3)

        # 3. Expand distances into RBF features
        edge_attr = self.distance_expansion(edge_weight)  # (E, num_rbf)

        # 4. Normalize direction vectors (skip zero-distance self-loops)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / edge_weight[mask].unsqueeze(1).clamp(min=1e-8)

        # 5. Initialize vector features to zero
        vec = torch.zeros(
            x.size(0), 3, x.size(1),
            device=x.device, dtype=x.dtype,
        )  # (N, 3, hidden)

        # 6. Equivariant attention layers with residual connections
        for attn_layer in self.attention_layers:
            dx, dvec = attn_layer(
                x, vec, edge_index, edge_weight, edge_attr, edge_vec
            )
            x = x + dx
            vec = vec + dvec

        # 7. Normalize scalar output
        x = self.out_norm(x)

        # 8. Project equivariant vectors to displacement: (N, 3, C) -> (N, 3)
        return self.vec_proj(vec).squeeze(-1)  # (N, 3)
