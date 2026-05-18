"""DCE — Doubly Calibrated Estimator skeleton (arXiv:2403.00817)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibratedPropensityModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 32, n_experts: int = 5):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.router = nn.Sequential(nn.Linear(emb_dim, n_experts), nn.Softmax(dim=1))
        self.a = nn.Parameter(torch.ones(n_experts))
        self.b = nn.Parameter(-torch.ones(n_experts))

    def forward(self, users: torch.Tensor, items: torch.Tensor, T: float = 1e-3) -> torch.Tensor:
        u = self.user_emb(users)
        v = self.item_emb(items)
        logit = (u * v).sum(-1)

        pi = self.router(u)
        g = -torch.log(-torch.log(torch.rand_like(pi) + 1e-10) + 1e-10)
        pi = F.softmax((pi.log() + g) / T, dim=1)

        logit_exp = logit.unsqueeze(1).expand(-1, self.a.size(0))
        p_cal = torch.sigmoid(logit_exp * self.a + self.b)
        return (p_cal * pi).sum(1).clamp(1e-4, 1 - 1e-4)


def dce_dr_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    prop: torch.Tensor,
    imp_pred: torch.Tensor,
    gamma: float = 0.05,
) -> torch.Tensor:
    inv_p = 1.0 / prop.detach().clamp(gamma, 1.0)
    ips_term = F.binary_cross_entropy(pred, label, weight=inv_p, reduction="mean")
    imp_term = F.binary_cross_entropy(pred, imp_pred.detach(), reduction="mean")
    return ips_term - imp_term


def main() -> None:
    n_users, n_items = 5000, 2000
    model = CalibratedPropensityModel(n_users, n_items, emb_dim=32, n_experts=5)

    users = torch.randint(0, n_users, (128,))
    items = torch.randint(0, n_items, (128,))
    labels = torch.randint(0, 2, (128,)).float()

    prop = model(users, items)
    print(f"校准倾向分均值: {prop.mean():.4f}, 标准差: {prop.std():.4f}")

    pred = torch.sigmoid(torch.randn(128))
    imp = torch.sigmoid(torch.randn(128))
    loss = dce_dr_loss(pred, labels, prop, imp)
    print(f"DR Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
