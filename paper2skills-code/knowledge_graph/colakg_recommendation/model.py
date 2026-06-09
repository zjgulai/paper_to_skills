"""CoLaKG skeleton (SIGIR 2025, arXiv:2410.12229)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGSemanticAggregator(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = nn.Linear(embed_dim * 2, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_src, h_dst = x[src], x[dst]
        alpha_raw = self.attn(torch.cat([h_dst, h_src], dim=-1)).squeeze(-1) * edge_weight

        alpha_exp = torch.exp(alpha_raw - alpha_raw.max())
        denom = torch.zeros(x.size(0), device=x.device).index_add(0, dst, alpha_exp) + 1e-10
        alpha = alpha_exp / denom[dst]

        messages = alpha.unsqueeze(-1) * h_src
        agg = torch.zeros_like(x).index_add(0, dst, messages)
        return agg


class CoLaKG(nn.Module):
    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64, llm_embed_dim: int = 768):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.semantic_proj = nn.Linear(llm_embed_dim, embed_dim)
        self.kg_agg = KGSemanticAggregator(embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        item_item_edge_index: torch.Tensor,
        item_item_weight: torch.Tensor,
        llm_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        s = self.semantic_proj(llm_embeddings)
        s_aug = self.kg_agg(s, item_item_edge_index, item_item_weight)

        e_v = self.item_emb.weight
        gate = self.gate(torch.cat([e_v, s_aug], dim=-1))
        item_repr = gate * e_v + (1 - gate) * s_aug

        u = self.user_emb(users)
        v = item_repr[items]
        return (u * v).sum(dim=-1)


def main() -> None:
    N_U, N_I, D, D_LLM = 500, 1000, 64, 768
    model = CoLaKG(N_U, N_I, D, D_LLM)
    edge_idx = torch.randint(0, N_I, (2, 5000))
    edge_w = torch.rand(5000)
    llm_embs = torch.randn(N_I, D_LLM)
    users = torch.randint(0, N_U, (32,))
    items = torch.randint(0, N_I, (32,))
    scores = model(users, items, edge_idx, edge_w, llm_embs)
    print(f"Output shape: {scores.shape}")
    print(f"Sample scores: {scores[:5].detach().tolist()}")


if __name__ == "__main__":
    main()
