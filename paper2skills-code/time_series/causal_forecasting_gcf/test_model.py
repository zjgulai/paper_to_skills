"""Smoke test for causal_forecasting_gcf."""
import torch

from .model import GCF, estimate_ate


def test_gcf_forward_shape():
    N, T, F = 10, 12, 16
    model = GCF(in_dim=F, hid_dim=32)
    x = torch.randn(N, T, F)
    edge_index = torch.randint(0, N, (2, 30))
    edge_type = torch.randint(0, 3, (30,))
    out = model(x, edge_index, edge_type)
    assert out.shape == (N, 1)


def test_estimate_ate_signature():
    y_obs = torch.tensor([100.0, 80.0, 120.0, 50.0])
    y_cf = torch.tensor([110.0, 90.0, 130.0, 60.0])
    treated = torch.tensor([True, True, False, False])
    ate = estimate_ate(y_obs, y_cf, treated)
    assert isinstance(ate, float)
    assert abs(ate - 10.0) < 1e-3


if __name__ == "__main__":
    test_gcf_forward_shape()
    test_estimate_ate_signature()
    print("OK")
