"""Smoke test for colakg_recommendation."""
import torch

from .model import CoLaKG


def test_colakg_forward_returns_scalar_per_pair():
    model = CoLaKG(n_users=20, n_items=50, embed_dim=8, llm_embed_dim=16)
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([3, 5, 7])
    edges = torch.randint(0, 50, (2, 100))
    weights = torch.rand(100)
    llm_embs = torch.randn(50, 16)
    scores = model(users, items, edges, weights, llm_embs)
    assert scores.shape == (3,)


if __name__ == "__main__":
    test_colakg_forward_returns_scalar_per_pair()
    print("OK")
