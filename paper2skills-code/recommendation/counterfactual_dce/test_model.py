"""Smoke test for counterfactual_dce."""
import torch

from .model import CalibratedPropensityModel, dce_dr_loss


def test_calibrated_propensity_in_range():
    model = CalibratedPropensityModel(n_users=100, n_items=50, emb_dim=8, n_experts=3)
    users = torch.tensor([0, 1, 2, 3])
    items = torch.tensor([10, 20, 30, 40])
    prop = model(users, items)
    assert prop.shape == (4,)
    assert torch.all(prop > 0) and torch.all(prop < 1)


def test_dce_dr_loss_runs():
    pred = torch.sigmoid(torch.randn(16))
    label = torch.randint(0, 2, (16,)).float()
    prop = torch.rand(16) * 0.5 + 0.25
    imp = torch.sigmoid(torch.randn(16))
    loss = dce_dr_loss(pred, label, prop, imp)
    assert loss.shape == torch.Size([])
    assert not torch.isnan(loss)


if __name__ == "__main__":
    test_calibrated_propensity_in_range()
    test_dce_dr_loss_runs()
    print("OK")
