"""GCF — Graph Causal Forecasting skeleton (AAAI 2025, Amazon, no public code)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNEncoder(nn.Module):
    def __init__(self, in_dim: int = 32, hid_dim: int = 64, n_relations: int = 3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_relations, in_dim, hid_dim) * 0.1)
        self.W_self = nn.Linear(in_dim, hid_dim)
        self.n_relations = n_relations

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = self.W_self(x)
        for r in range(self.n_relations):
            mask = edge_type == r
            if mask.sum() == 0:
                continue
            src = edge_index[0][mask]
            dst = edge_index[1][mask]
            messages = x[src] @ self.W[r]
            h = h.index_add(0, dst, messages)
        return F.relu(h)


class DilatedTCN(nn.Module):
    def __init__(self, hid_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(hid_dim, hid_dim, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(hid_dim, hid_dim, kernel_size=3, dilation=2, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(hid_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        return self.out(h)


class GCF(nn.Module):
    def __init__(self, in_dim: int = 32, hid_dim: int = 64):
        super().__init__()
        self.encoder = RGCNEncoder(in_dim, hid_dim)
        self.decoder = DilatedTCN(hid_dim)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        T = x_seq.shape[1]
        h_list = [self.encoder(x_seq[:, t, :], edge_index, edge_type) for t in range(T)]
        h_seq = torch.stack(h_list, dim=2)
        return self.decoder(h_seq)


def estimate_ate(y_factual: torch.Tensor, y_counterfactual: torch.Tensor, treated_mask: torch.Tensor) -> float:
    ate = (y_counterfactual[treated_mask] - y_factual[treated_mask]).mean()
    return float(ate.item())


def main() -> None:
    N, T, F_dim = 20, 30, 32
    x = torch.randn(N, T, F_dim)
    edge_index = torch.randint(0, N, (2, 60))
    edge_type = torch.randint(0, 3, (60,))
    treated = torch.zeros(N, dtype=torch.bool)
    treated[:5] = True

    model = GCF(in_dim=F_dim)
    y_cf = model(x, edge_index, edge_type).squeeze()
    y_obs = torch.randn(N) * 50 + 200

    ate = estimate_ate(y_obs, y_cf, treated)
    print(f"反事实需求差(ATE) = {ate:.2f} 件/天")


if __name__ == "__main__":
    main()
