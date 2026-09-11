"""
NeuralNDCG: Learning to Rank with Differentiable Sorting
基于可微分排序松弛的NDCG直接优化

论文: NeuralNDCG: Direct Optimisation of a Ranking Metric via
      Differentiable Relaxation of Sorting
arXiv: 2102.07831
代码: https://github.com/allegro/allRank
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


# ============ 1. NeuralSort: 可微分排序松弛 ============

def neural_sort(scores: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    NeuralSort: 用softmax生成近似排序分配矩阵

    P_hat[b, i, j] = 第j个item排在第i位的概率
    """
    batch_size, list_size = scores.shape

    # 计算每个 item j 的 pairwise 绝对差和: sum_k |s_j - s_k|
    scores_j = scores.unsqueeze(1)  # [B, 1, N]
    scores_k = scores.unsqueeze(2)  # [B, N, 1]
    abs_diff = torch.abs(scores_j - scores_k)  # [B, N, N]
    abs_diff_sum = abs_diff.sum(dim=2, keepdim=True)  # [B, N, 1]

    # row_factor[i] = n + 1 - 2i
    row_indices = torch.arange(1, list_size + 1, device=scores.device, dtype=scores.dtype)
    row_factor = (list_size + 1 - 2 * row_indices)  # [N]

    # term1[b, i, j] = row_factor[i] * scores[b, j]
    term1 = row_factor.view(1, -1, 1) * scores.unsqueeze(1)  # [B, N, N]

    sorted_scores = term1 - abs_diff_sum  # [B, N, N]
    P_hat = F.softmax(sorted_scores / tau, dim=2)
    return P_hat


def sinkhorn_scaling(P: torch.Tensor, num_iter: int = 30,
                     epsilon: float = 1e-6) -> torch.Tensor:
    """Sinkhorn迭代归一化：使矩阵行列和均为1"""
    for _ in range(num_iter):
        P = P / (P.sum(dim=2, keepdim=True) + epsilon)
        P = P / (P.sum(dim=1, keepdim=True) + epsilon)
    return P


# ============ 2. NDCG相关函数 ============

def ndcg_at_k(scores: torch.Tensor, relevances: torch.Tensor,
              k: Optional[int] = None) -> torch.Tensor:
    """计算标准NDCG@k（不可微，仅用于评估）"""
    if k is None:
        k = scores.shape[1]
    _, sorted_indices = torch.sort(scores, dim=1, descending=True)
    sorted_rels = torch.gather(relevances, 1, sorted_indices)[:, :k]
    positions = torch.arange(1, k + 1, device=scores.device, dtype=torch.float32)
    discounts = 1.0 / torch.log2(positions + 1)
    gains = (2.0 ** sorted_rels - 1.0)
    dcg = (gains * discounts).sum(dim=1)
    ideal_sorted = torch.sort(relevances, dim=1, descending=True)[0][:, :k]
    ideal_gains = (2.0 ** ideal_sorted - 1.0)
    max_dcg = (ideal_gains * discounts).sum(dim=1)
    max_dcg = torch.where(max_dcg == 0, torch.ones_like(max_dcg), max_dcg)
    return dcg / max_dcg


# ============ 3. NeuralNDCG损失 ============

class NeuralNDCGLoss(nn.Module):
    """NeuralNDCG损失：直接优化NDCG的可微分近似"""

    def __init__(self, k: Optional[int] = None, tau: float = 1.0,
                 num_sinkhorn_iter: int = 30, variant: str = "standard"):
        super().__init__()
        self.k = k
        self.tau = tau
        self.num_sinkhorn_iter = num_sinkhorn_iter
        self.variant = variant

    def forward(self, scores: torch.Tensor, relevances: torch.Tensor) -> torch.Tensor:
        batch_size, list_size = scores.shape
        k = self.k or list_size

        P_hat = neural_sort(scores, self.tau)
        P_hat = sinkhorn_scaling(P_hat, self.num_sinkhorn_iter)
        gains = (2.0 ** relevances - 1.0).unsqueeze(2)

        if self.variant == "standard":
            sorted_gains = torch.bmm(P_hat, gains).squeeze(2)
            positions = torch.arange(1, list_size + 1,
                                     device=scores.device, dtype=scores.dtype)
            discounts = 1.0 / torch.log2(positions + 1)
            dcg = (sorted_gains[:, :k] * discounts[:k]).sum(dim=1)
        else:
            positions = torch.arange(1, list_size + 1,
                                     device=scores.device, dtype=scores.dtype)
            discounts = 1.0 / torch.log2(positions + 1)
            discounts_k = discounts.clone()
            discounts_k[k:] = 0
            weighted_discounts = torch.bmm(
                P_hat.transpose(1, 2),
                discounts_k.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1)
            ).squeeze(2)
            dcg = (gains.squeeze(2) * weighted_discounts).sum(dim=1)

        ideal_sorted = torch.sort(relevances, dim=1, descending=True)[0]
        ideal_positions = torch.arange(1, list_size + 1, device=scores.device, dtype=torch.float32)
        ideal_discounts = 1.0 / torch.log2(ideal_positions + 1)
        ideal_gains = (2.0 ** ideal_sorted - 1.0)
        max_dcg = (ideal_gains * ideal_discounts).sum(dim=1)
        max_dcg = torch.where(max_dcg == 0, torch.ones_like(max_dcg), max_dcg)
        neural_ndcg = dcg / max_dcg
        return -neural_ndcg.mean()


# ============ 4. 三种LTR损失对比 ============

class PointwiseMSELoss(nn.Module):
    """Pointwise: MSE回归损失"""
    def forward(self, scores: torch.Tensor, relevances: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(scores, relevances.float())


class PairwiseRankNetLoss(nn.Module):
    """Pairwise: RankNet损失"""
    def forward(self, scores: torch.Tensor, relevances: torch.Tensor) -> torch.Tensor:
        scores_i = scores.unsqueeze(2)
        scores_j = scores.unsqueeze(1)
        rels_i = relevances.unsqueeze(2)
        rels_j = relevances.unsqueeze(1)
        S = torch.sign(rels_i - rels_j).float()
        mask = (S != 0).float()
        sigma = 1.0
        diff = sigma * (scores_i - scores_j)
        loss = -S * diff + torch.log1p(torch.exp(diff))
        return (loss * mask).sum() / (mask.sum() + 1e-8)


# ============ 5. 评分模型 ============

class ScoringModel(nn.Module):
    """简单MLP评分模型"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, list_size, feat_dim = features.shape
        features_flat = features.view(-1, feat_dim)
        scores_flat = self.network(features_flat).squeeze(-1)
        return scores_flat.view(batch_size, list_size)


# ============ 6. 测试用例 ============

def create_synthetic_data(num_queries: int = 100, list_size: int = 20,
                          feat_dim: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成合成LTR数据"""
    np.random.seed(42)
    torch.manual_seed(42)
    features = torch.randn(num_queries, list_size, feat_dim)
    rel_scores = features[:, :, 0] + 0.5 * torch.randn(num_queries, list_size)
    relevances = torch.clamp((rel_scores + 2).long(), 0, 4)
    return features, relevances


def test_three_paradigms():
    """对比三种LTR范式的性能"""
    print("=" * 60)
    print("三种LTR范式对比测试")
    print("=" * 60)

    train_features, train_rels = create_synthetic_data(num_queries=200)
    val_features, val_rels = create_synthetic_data(num_queries=50)
    feat_dim = train_features.shape[2]

    losses = {
        "Pointwise (MSE)": PointwiseMSELoss(),
        "Pairwise (RankNet)": PairwiseRankNetLoss(),
        "Listwise (NeuralNDCG)": NeuralNDCGLoss(k=10, tau=1.0, variant="standard"),
    }

    results = {}
    for name, loss_fn in losses.items():
        print(f"\n--- {name} ---")
        model = ScoringModel(feat_dim, hidden_dims=[64, 32])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(50):
            model.train()
            scores = model(train_features)
            loss = loss_fn(scores, train_rels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            val_scores = model(val_features)
            ndcg = ndcg_at_k(val_scores, val_rels, k=10)
            mean_ndcg = ndcg.mean().item()
            results[name] = mean_ndcg
            print(f"  Val NDCG@10: {mean_ndcg:.4f}")

    print("\n" + "=" * 60)
    print("结果对比:")
    for name, ndcg in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {ndcg:.4f}")
    print("=" * 60)
    return results


def test_neural_ndcg_variants():
    """对比NeuralNDCG两种变体"""
    print("\n" + "=" * 60)
    print("NeuralNDCG 变体对比")
    print("=" * 60)

    train_features, train_rels = create_synthetic_data(num_queries=200)
    val_features, val_rels = create_synthetic_data(num_queries=50)
    feat_dim = train_features.shape[2]

    variants = {
        "Standard (tau=1.0)": NeuralNDCGLoss(k=10, tau=1.0, variant="standard"),
        "Standard (tau=2.0)": NeuralNDCGLoss(k=10, tau=2.0, variant="standard"),
        "Transposed (tau=1.0)": NeuralNDCGLoss(k=10, tau=1.0, variant="transposed"),
    }

    for name, loss_fn in variants.items():
        model = ScoringModel(feat_dim, hidden_dims=[64, 32])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(50):
            model.train()
            scores = model(train_features)
            loss = loss_fn(scores, train_rels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            val_scores = model(val_features)
            ndcg = ndcg_at_k(val_scores, val_rels, k=10)
            print(f"  {name}: NDCG@10 = {ndcg.mean().item():.4f}")


if __name__ == "__main__":
    test_three_paradigms()
    test_neural_ndcg_variants()
