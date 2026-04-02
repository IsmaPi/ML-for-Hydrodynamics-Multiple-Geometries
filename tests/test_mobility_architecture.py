"""Tests for the mobility tensor architecture.

Verifies all models (ET, TN, GN) satisfy key physics properties:
linearity in forces, rotational equivariance, correct shape, zero-init.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import ModelConfig
from models.torchmd_et import HydroTorchMD_ET
from models.torchmd_gn import HydroTorchMD_GN
from models.torchmd_tn import HydroTorchMD_TN
from models.mobility_head import MobilityHead


def _make_cfg(model_type):
    return ModelConfig(
        model_type=model_type,
        hidden_dim=32,
        num_layers=1,
        cutoff_radius=15.0,
        max_num_neighbors=32,
        num_rbf=8,
        rbf_type="expnorm",
        trainable_rbf=False,
        num_heads=2,
        distance_influence="both",
        mobility_hidden_dim=16,
    )


def _make_data(N=6, seed=42):
    rng = torch.manual_seed(seed)
    positions = torch.randn(N, 3) * 5.0
    for i in range(1, N):
        while (positions[i] - positions[:i]).norm(dim=-1).min() < 2.5:
            positions[i] = torch.randn(3) * 5.0
    return Data(
        x=torch.ones(N, dtype=torch.long),
        pos=positions,
        forces=torch.randn(N, 3),
        y=torch.randn(N, 3),
        num_nodes=N,
        batch=torch.zeros(N, dtype=torch.long),
    )


def _random_rotation():
    q = torch.randn(4)
    q = q / q.norm()
    w, x, y, z = q
    return torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


MODELS = {
    "torchmd_et": HydroTorchMD_ET,
    "torchmd_gn": HydroTorchMD_GN,
    "torchmd_tn": HydroTorchMD_TN,
}


@pytest.fixture(params=["torchmd_et", "torchmd_gn", "torchmd_tn"])
def model(request):
    cfg = _make_cfg(request.param)
    m = MODELS[request.param](cfg)
    m.eval()
    return m


def test_output_shape(model):
    for N in [2, 4, 16]:
        data = _make_data(N=N)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (N, 3)


def test_linearity_scaling(model):
    data = _make_data()
    with torch.no_grad():
        out1 = model(data)
        data2 = data.clone()
        data2.forces = data.forces * 2.5
        out2 = model(data2)
    torch.testing.assert_close(out2, 2.5 * out1, atol=1e-4, rtol=1e-4)


def test_linearity_additivity(model):
    data = _make_data()
    F1, F2 = torch.randn_like(data.forces), torch.randn_like(data.forces)
    with torch.no_grad():
        d1 = data.clone(); d1.forces = F1
        d2 = data.clone(); d2.forces = F2
        d12 = data.clone(); d12.forces = F1 + F2
        torch.testing.assert_close(model(d12), model(d1) + model(d2), atol=1e-4, rtol=1e-4)


def test_equivariance(model):
    data = _make_data()
    R = _random_rotation()
    with torch.no_grad():
        out_orig = model(data)
        data_rot = data.clone()
        data_rot.pos = (R @ data.pos.T).T
        data_rot.forces = (R @ data.forces.T).T
        out_rot = model(data_rot)
    torch.testing.assert_close(out_rot, (R @ out_orig.T).T, atol=1e-3, rtol=1e-3)


def test_zero_init_is_self_mobility(model):
    """At init, edge MLP is zero so output should be self_mobility * forces."""
    data = _make_data()
    with torch.no_grad():
        out = model(data)
    # self_mobility defaults to 1.0 with default viscosity=1/(6*pi), a=1
    expected = data.forces  # self_mobility * F = 1.0 * F
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


def test_mobility_head_zero_init_is_self_mobility():
    head = MobilityHead(node_dim=16, hidden_dim=8, cutoff_upper=10.0)
    N = 4
    edge_index = torch.tensor([[0,1,2,3],[1,2,3,0]], dtype=torch.long)
    edge_weight = torch.full((4,), 3.0)
    edge_vec = torch.randn(4, 3)
    forces = torch.randn(N, 3)
    with torch.no_grad():
        out = head(torch.randn(N, 16), edge_index, edge_weight, edge_vec, forces)
    # Edge MLP is zero-init, so output = self_mobility * forces
    expected = head.self_mobility * forces
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_mobility_head_cutoff():
    head = MobilityHead(node_dim=16, hidden_dim=8, cutoff_upper=5.0)
    with torch.no_grad():
        head.edge_mlp[-1].weight.fill_(1.0)
        head.edge_mlp[-1].bias.fill_(1.0)
    N = 2
    forces = torch.randn(N, 3)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_weight = torch.tensor([100.0])  # far beyond cutoff
    edge_vec = torch.tensor([[100.0, 0.0, 0.0]])
    with torch.no_grad():
        out = head(torch.randn(N, 16), edge_index, edge_weight, edge_vec, forces)
    # Node 1 should only have self-mobility, no interaction from distant node 0
    expected_1 = head.self_mobility * forces[1]
    torch.testing.assert_close(out[1], expected_1, atol=1e-6, rtol=1e-6)
