"""
CAGED: Causality-aware Graph Aggregation Weight Estimator for Debiasing
论文: Causality-aware Graph Aggregation Weight Estimator for Popularity Debiasing in Top-K Recommendation
arXiv: 2510.04502 (2025-10)

核心思想:
  - GNN 聚合权重 ≈ 观测数据的交互似然分布（带流行度偏差）
  - 后门调整（Backdoor Adjustment）：通过 Encoder-Decoder (ELBO优化) 学到无偏聚合权重
  - 动量更新（Momentum Update）：早期训练平滑过渡，逐步注入无偏权重
  - 将无偏权重代回 LightGCN 做图传播

业务场景: 跨境电商长尾商品挖掘 / DTC 独立站新品冷启动
代码结构: Mock 数据 → CAGED 核心模块 → 无偏权重估计 → LightGCN 传播 → 推荐打分 → 自测
"""

import math
import unittest
import numpy as np
from typing import Tuple, Dict, List


# ─────────────────────────────────────────────
# 1. 工具函数
# ─────────────────────────────────────────────

def compute_bipartite_adj(
    interactions: List[Tuple[int, int]],
    num_users: int,
    num_items: int,
) -> np.ndarray:
    """
    构建用户-商品二部图邻接矩阵（未归一化）。
    interactions: [(user_id, item_id), ...]
    返回 shape (num_users+num_items, num_users+num_items) 的对称邻接矩阵
    """
    N = num_users + num_items
    adj = np.zeros((N, N), dtype=np.float32)
    for u, i in interactions:
        adj[u, num_users + i] = 1.0
        adj[num_users + i, u] = 1.0
    return adj


def degree_norm_weights(adj: np.ndarray) -> np.ndarray:
    """
    标准 LightGCN 聚合权重：D^{-1/2} A D^{-1/2}
    这等价于使用节点度数的平方根作为聚合权重。
    论文指出：此权重隐含了观测数据的流行度先验（带偏）。
    """
    degree = adj.sum(axis=1)
    safe_degree = np.where(degree > 0, degree, 1.0)
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(safe_degree), 0.0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    return norm_adj


# ─────────────────────────────────────────────
# 2. CAGED 无偏权重估计器（Encoder-Decoder, ELBO）
# ─────────────────────────────────────────────

class CAGEDWeightEstimator:
    """
    CAGED 核心模块：通过 Encoder-Decoder 优化 ELBO 估计无偏聚合权重。

    论文关键推导（简化）:
      - 观测权重 w_ij = p(r_ij=1 | e_u, e_i) = σ(u·i / sqrt(pop_i))
        其中 pop_i 是商品 i 的流行度（交互数）
      - 无偏权重 w*_ij 通过去除流行度混淆变量来估计
      - ELBO 目标: E[log p(r|z)] - KL[q(z|r) || p(z)]
        其中 z 是潜在的"真实偏好强度"

    本实现用 Mock 版本：
      - Encoder: 从交互矩阵估计潜在偏好分布 q(z|r)
      - Decoder: 从 z 重建交互 p(r|z)
      - 无偏权重 = Decoder 输出经过流行度校正后的权重
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        latent_dim: int = 16,
        momentum: float = 0.9,
        random_seed: int = 42,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.momentum = momentum  # 动量更新系数 (早期训练平滑用)
        rng = np.random.RandomState(random_seed)

        # 用户和商品潜在表征（随机初始化，实际应通过训练优化）
        self.user_mu = rng.randn(num_users, latent_dim).astype(np.float32) * 0.1
        self.user_logvar = np.zeros((num_users, latent_dim), dtype=np.float32) - 2.0

        self.item_mu = rng.randn(num_items, latent_dim).astype(np.float32) * 0.1
        self.item_logvar = np.zeros((num_items, latent_dim), dtype=np.float32) - 2.0

        # 商品流行度（交互计数），用于后门调整
        self.item_popularity = np.zeros(num_items, dtype=np.float32)

        # 动量权重矩阵（用于平滑过渡）
        self._momentum_weights = None

    def fit(
        self,
        interactions: List[Tuple[int, int]],
        epochs: int = 10,
        lr: float = 0.01,
    ) -> Dict[str, List[float]]:
        """
        训练 CAGED 估计器，最小化 -ELBO。
        返回训练日志 {'elbo': [...], 'recon_loss': [...], 'kl_loss': [...]}
        """
        # 统计流行度
        self.item_popularity = np.zeros(self.num_items, dtype=np.float32)
        for _, item_id in interactions:
            self.item_popularity[item_id] += 1.0
        # 归一化到 [0, 1]
        max_pop = self.item_popularity.max() + 1e-8
        self.item_popularity /= max_pop

        history = {"elbo": [], "recon_loss": [], "kl_loss": []}

        # 简化训练循环（实际应使用 PyTorch autograd）
        for epoch in range(epochs):
            total_recon = 0.0
            total_kl = 0.0

            for u, i in interactions:
                # --- Encoder: 采样潜在变量 z ---
                z_u = self._reparameterize(self.user_mu[u], self.user_logvar[u])
                z_i = self._reparameterize(self.item_mu[i], self.item_logvar[i])

                # --- Decoder: 重建交互概率 ---
                recon_prob = self._sigmoid(np.dot(z_u, z_i))
                recon_loss = -np.log(recon_prob + 1e-8)

                # --- KL 散度（解析解）---
                kl_u = self._kl_normal(self.user_mu[u], self.user_logvar[u])
                kl_i = self._kl_normal(self.item_mu[i], self.item_logvar[i])
                kl_loss = (kl_u + kl_i) / len(interactions)

                total_recon += recon_loss
                total_kl += kl_loss

                # --- 简化梯度更新（近似）---
                grad_scale = lr * (1.0 - self.item_popularity[i])  # 长尾商品梯度更大
                self.user_mu[u] += grad_scale * z_i * (1 - recon_prob)
                self.item_mu[i] += grad_scale * z_u * (1 - recon_prob)

            elbo = -(total_recon + total_kl) / len(interactions)
            history["elbo"].append(float(elbo))
            history["recon_loss"].append(float(total_recon / len(interactions)))
            history["kl_loss"].append(float(total_kl / len(interactions)))

        return history

    def compute_unbiased_weights(
        self,
        biased_adj: np.ndarray,
        epoch: int = 0,
        total_epochs: int = 10,
    ) -> np.ndarray:
        """
        计算无偏聚合权重矩阵（后门调整版）。

        关键步骤:
          1. 计算每条边的"因果偏好强度"（去除流行度混淆）
          2. 用动量更新策略平滑融合有偏/无偏权重

        动量策略: w_unbiased_t = α * w_biased + (1-α) * w_causal
          其中 α = momentum^(epoch/total_epochs) 在训练中逐步降低
        """
        N = self.num_users + self.num_items
        causal_adj = np.zeros_like(biased_adj)

        # 计算后门调整权重（用因果表征点积代替度数归一化）
        for u in range(self.num_users):
            for i in range(self.num_items):
                edge_weight = biased_adj[u, self.num_users + i]
                if edge_weight > 0:
                    z_u = self.user_mu[u]
                    z_i = self.item_mu[i]
                    # 因果偏好强度（去除流行度压制）
                    causal_score = self._sigmoid(np.dot(z_u, z_i))
                    # 流行度校正：长尾商品权重提升
                    pop_correction = 1.0 / (1.0 + self.item_popularity[i])
                    causal_adj[u, self.num_users + i] = causal_score * pop_correction
                    causal_adj[self.num_users + i, u] = causal_score * pop_correction

        # 行归一化（让权重矩阵总和为1）
        row_sum = causal_adj.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        causal_adj = causal_adj / row_sum

        # 动量平滑过渡
        alpha = self.momentum ** (epoch / max(total_epochs, 1))
        if self._momentum_weights is None:
            self._momentum_weights = biased_adj.copy()

        unbiased_weights = alpha * self._momentum_weights + (1.0 - alpha) * causal_adj
        self._momentum_weights = unbiased_weights  # 更新动量缓存

        return unbiased_weights

    @staticmethod
    def _reparameterize(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """重参数化技巧：z = mu + eps * exp(0.5 * logvar)"""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape).astype(np.float32) * 0.01
        return mu + eps * std

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-float(np.clip(x, -30, 30))))

    @staticmethod
    def _kl_normal(mu: np.ndarray, logvar: np.ndarray) -> float:
        """KL(N(mu, var) || N(0, 1)) 解析解"""
        return float(0.5 * np.sum(np.exp(logvar) + mu ** 2 - 1.0 - logvar))


# ─────────────────────────────────────────────
# 3. LightGCN（带无偏权重注入）
# ─────────────────────────────────────────────

class LightGCNWithCAGED:
    """
    LightGCN 推荐模型，支持注入 CAGED 无偏聚合权重。

    核心改动: 图传播时使用 CAGED 输出的无偏权重，而非标准度数归一化权重。
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 16,
        n_layers: int = 3,
        random_seed: int = 42,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        rng = np.random.RandomState(random_seed)
        N = num_users + num_items
        # 节点初始嵌入
        self.embeddings = rng.randn(N, embed_dim).astype(np.float32) * 0.1

    def propagate(self, adj_weight: np.ndarray) -> np.ndarray:
        """
        图传播：E^(k) = A_unbiased * E^(k-1)
        最终嵌入 = 各层嵌入的均值（LightGCN 原论文设定）
        """
        all_embeddings = [self.embeddings.copy()]
        current = self.embeddings.copy()

        for _ in range(self.n_layers):
            current = adj_weight @ current
            all_embeddings.append(current.copy())

        # 各层均值聚合
        final_embeddings = np.mean(all_embeddings, axis=0)
        return final_embeddings

    def recommend(
        self,
        embeddings: np.ndarray,
        user_id: int,
        top_k: int = 5,
        exclude_items: List[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        对指定用户给出 Top-K 推荐列表。
        返回 [(item_id, score), ...] 按 score 降序
        """
        u_emb = embeddings[user_id]
        exclude_set = set(exclude_items or [])

        scores = []
        for i in range(self.num_items):
            if i in exclude_set:
                continue
            i_emb = embeddings[self.num_users + i]
            score = float(np.dot(u_emb, i_emb))
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ─────────────────────────────────────────────
# 4. CAGED 推荐管线（完整端到端）
# ─────────────────────────────────────────────

class CAGEDRecommendationPipeline:
    """
    CAGED 完整推荐管线：
      1. 构建用户-商品二部图
      2. 训练 CAGED 估计无偏权重
      3. 注入无偏权重到 LightGCN
      4. 图传播 + 打分 + Top-K 推荐

    业务用法: 母婴电商长尾商品挖掘
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 16,
        latent_dim: int = 16,
        n_layers: int = 3,
        momentum: float = 0.9,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.caged = CAGEDWeightEstimator(
            num_users=num_users,
            num_items=num_items,
            latent_dim=latent_dim,
            momentum=momentum,
        )
        self.lightgcn = LightGCNWithCAGED(
            num_users=num_users,
            num_items=num_items,
            embed_dim=embed_dim,
            n_layers=n_layers,
        )
        self._biased_adj = None
        self._biased_norm = None

    def build_graph(self, interactions: List[Tuple[int, int]]) -> None:
        """构建二部图和标准度数归一化权重"""
        self._biased_adj = compute_bipartite_adj(
            interactions, self.num_users, self.num_items
        )
        self._biased_norm = degree_norm_weights(self._biased_adj)

    def train(
        self,
        interactions: List[Tuple[int, int]],
        epochs: int = 5,
    ) -> Dict[str, List[float]]:
        """训练 CAGED 估计器"""
        return self.caged.fit(interactions, epochs=epochs)

    def get_recommendations(
        self,
        user_id: int,
        top_k: int = 5,
        epoch: int = 4,
        total_epochs: int = 5,
        exclude_items: List[int] = None,
        use_unbiased: bool = True,
    ) -> List[Tuple[int, float]]:
        """获取推荐列表（可切换有偏/无偏权重）"""
        if use_unbiased:
            adj_weight = self.caged.compute_unbiased_weights(
                self._biased_norm,
                epoch=epoch,
                total_epochs=total_epochs,
            )
        else:
            adj_weight = self._biased_norm

        embeddings = self.lightgcn.propagate(adj_weight)
        return self.lightgcn.recommend(embeddings, user_id, top_k, exclude_items)

    def compare_biased_vs_unbiased(
        self,
        user_id: int,
        top_k: int = 5,
    ) -> Dict[str, List[Tuple[int, float]]]:
        """对比有偏推荐 vs 无偏推荐结果"""
        biased = self.get_recommendations(user_id, top_k=top_k, use_unbiased=False)
        unbiased = self.get_recommendations(user_id, top_k=top_k, use_unbiased=True)
        return {"biased": biased, "unbiased": unbiased}


# ─────────────────────────────────────────────
# 5. 评估工具
# ─────────────────────────────────────────────

def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """计算 NDCG@K"""
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    dcg = 0.0
    for idx, item in enumerate(recommended_k):
        if item in relevant_set:
            dcg += 1.0 / math.log2(idx + 2)

    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def long_tail_coverage(
    recommended_items: List[int],
    item_popularity: np.ndarray,
    tail_threshold: float = 0.2,
) -> float:
    """
    计算长尾商品覆盖率：推荐列表中流行度低于阈值的商品比例
    tail_threshold: 流行度归一化后低于此值视为长尾
    """
    tail_count = sum(
        1 for i in recommended_items if item_popularity[i] < tail_threshold
    )
    return tail_count / len(recommended_items) if recommended_items else 0.0


# ─────────────────────────────────────────────
# 6. 自测（unittest）
# ─────────────────────────────────────────────

class TestCAGED(unittest.TestCase):

    def setUp(self):
        """构造 Mock 数据：5 用户 x 8 商品，模拟流行度偏差"""
        self.num_users = 5
        self.num_items = 8

        # 模拟流行度偏差：商品 0、1 是"爆款"（大量交互），商品 5、6、7 是长尾
        self.interactions = [
            # 爆款商品 0 和 1（所有用户都交互）
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            # 商品 2、3（中等热度）
            (0, 2), (1, 2), (2, 3), (3, 3),
            # 长尾商品 5、6、7（少量用户交互）
            (0, 5), (1, 6), (2, 7),
        ]

        self.pipeline = CAGEDRecommendationPipeline(
            num_users=self.num_users,
            num_items=self.num_items,
            embed_dim=8,
            latent_dim=8,
            n_layers=2,
            momentum=0.8,
        )
        self.pipeline.build_graph(self.interactions)

    def test_bipartite_adj_shape(self):
        """测试二部图邻接矩阵形状正确"""
        adj = compute_bipartite_adj(
            self.interactions, self.num_users, self.num_items
        )
        expected_N = self.num_users + self.num_items
        self.assertEqual(adj.shape, (expected_N, expected_N))

    def test_adj_symmetry(self):
        """测试邻接矩阵对称性"""
        adj = compute_bipartite_adj(
            self.interactions, self.num_users, self.num_items
        )
        np.testing.assert_array_equal(adj, adj.T)

    def test_degree_norm_weights_no_nan(self):
        """测试度数归一化权重无 NaN/Inf"""
        adj = compute_bipartite_adj(
            self.interactions, self.num_users, self.num_items
        )
        norm_adj = degree_norm_weights(adj)
        self.assertFalse(np.any(np.isnan(norm_adj)))
        self.assertFalse(np.any(np.isinf(norm_adj)))

    def test_caged_fit_elbo_changes(self):
        """测试 CAGED 训练后 ELBO 有变化（说明模型在学习）"""
        history = self.pipeline.train(self.interactions, epochs=3)
        self.assertIn("elbo", history)
        self.assertEqual(len(history["elbo"]), 3)
        # ELBO 列表非空且不全相同（模型在更新）
        self.assertGreater(len(set([round(v, 6) for v in history["elbo"]])), 1)

    def test_caged_item_popularity_set(self):
        """测试流行度统计正确：爆款商品（0,1）流行度 > 长尾商品（5,6,7）"""
        self.pipeline.train(self.interactions, epochs=2)
        pop = self.pipeline.caged.item_popularity
        # 商品 0 和 1 各有 5 次交互（爆款），商品 5/6/7 各只有 1 次（长尾）
        self.assertGreater(pop[0], pop[5])
        self.assertGreater(pop[1], pop[6])

    def test_unbiased_weights_shape(self):
        """测试无偏权重矩阵形状正确"""
        self.pipeline.train(self.interactions, epochs=2)
        N = self.num_users + self.num_items
        norm_adj = self.pipeline._biased_norm
        unbiased_w = self.pipeline.caged.compute_unbiased_weights(norm_adj, epoch=1)
        self.assertEqual(unbiased_w.shape, (N, N))

    def test_unbiased_weights_no_nan(self):
        """测试无偏权重矩阵无 NaN/Inf"""
        self.pipeline.train(self.interactions, epochs=2)
        norm_adj = self.pipeline._biased_norm
        unbiased_w = self.pipeline.caged.compute_unbiased_weights(norm_adj, epoch=1)
        self.assertFalse(np.any(np.isnan(unbiased_w)))
        self.assertFalse(np.any(np.isinf(unbiased_w)))

    def test_lightgcn_propagation_shape(self):
        """测试 LightGCN 图传播输出形状正确"""
        N = self.num_users + self.num_items
        adj = self.pipeline._biased_norm
        embeddings = self.pipeline.lightgcn.propagate(adj)
        self.assertEqual(embeddings.shape, (N, 8))

    def test_recommend_returns_top_k(self):
        """测试推荐函数返回 Top-K 商品"""
        self.pipeline.train(self.interactions, epochs=2)
        recs = self.pipeline.get_recommendations(user_id=0, top_k=3)
        self.assertEqual(len(recs), 3)
        # 每条推荐是 (item_id, score) 元组
        for item_id, score in recs:
            self.assertIsInstance(item_id, int)
            self.assertIsInstance(score, float)

    def test_recommend_scores_sorted_descending(self):
        """测试推荐列表按 score 降序排列"""
        self.pipeline.train(self.interactions, epochs=2)
        recs = self.pipeline.get_recommendations(user_id=1, top_k=5)
        scores = [s for _, s in recs]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_exclude_items_honored(self):
        """测试排除已交互商品功能"""
        self.pipeline.train(self.interactions, epochs=2)
        # 用户 0 已交互：商品 0, 1, 2, 5
        exclude = [0, 1, 2, 5]
        recs = self.pipeline.get_recommendations(
            user_id=0, top_k=3, exclude_items=exclude
        )
        rec_items = [item_id for item_id, _ in recs]
        for item in exclude:
            self.assertNotIn(item, rec_items)

    def test_ndcg_at_k_perfect(self):
        """测试 NDCG@K 完美推荐场景"""
        ndcg = ndcg_at_k(recommended=[0, 1, 2], relevant=[0, 1, 2], k=3)
        self.assertAlmostEqual(ndcg, 1.0, places=5)

    def test_ndcg_at_k_empty(self):
        """测试 NDCG@K 无相关商品场景"""
        ndcg = ndcg_at_k(recommended=[3, 4, 5], relevant=[], k=3)
        self.assertEqual(ndcg, 0.0)

    def test_long_tail_coverage_unbiased_vs_biased(self):
        """
        核心业务测试：无偏推荐的长尾覆盖率 >= 有偏推荐的长尾覆盖率。
        体现 CAGED 对长尾商品的挖掘能力。
        """
        self.pipeline.train(self.interactions, epochs=5)
        pop = self.pipeline.caged.item_popularity

        biased_recs = self.pipeline.get_recommendations(
            user_id=2, top_k=4, use_unbiased=False
        )
        unbiased_recs = self.pipeline.get_recommendations(
            user_id=2, top_k=4, use_unbiased=True
        )

        biased_items = [i for i, _ in biased_recs]
        unbiased_items = [i for i, _ in unbiased_recs]

        tail_biased = long_tail_coverage(biased_items, pop, tail_threshold=0.3)
        tail_unbiased = long_tail_coverage(unbiased_items, pop, tail_threshold=0.3)

        # 打印对比结果（方便人工复核）
        print(f"\n[长尾覆盖率对比] 有偏: {tail_biased:.2f} | 无偏: {tail_unbiased:.2f}")
        print(f"  有偏推荐: {biased_items}")
        print(f"  无偏推荐: {unbiased_items}")
        print(f"  流行度:   {[round(float(pop[i]),2) for i in range(self.num_items)]}")

        # 无偏推荐的长尾覆盖率不低于有偏推荐（CAGED 核心效果验证）
        self.assertGreaterEqual(tail_unbiased, tail_biased)

    def test_compare_biased_vs_unbiased_output_structure(self):
        """测试 compare_biased_vs_unbiased 输出格式正确"""
        self.pipeline.train(self.interactions, epochs=2)
        result = self.pipeline.compare_biased_vs_unbiased(user_id=0, top_k=3)
        self.assertIn("biased", result)
        self.assertIn("unbiased", result)
        self.assertEqual(len(result["biased"]), 3)
        self.assertEqual(len(result["unbiased"]), 3)

    def test_momentum_update_smoothing(self):
        """测试动量更新：早期 epoch 有偏权重占比更高"""
        self.pipeline.train(self.interactions, epochs=5)
        norm_adj = self.pipeline._biased_norm

        # 重置动量缓存
        self.pipeline.caged._momentum_weights = None
        w_early = self.pipeline.caged.compute_unbiased_weights(
            norm_adj, epoch=0, total_epochs=10
        )
        self.pipeline.caged._momentum_weights = None
        w_late = self.pipeline.caged.compute_unbiased_weights(
            norm_adj, epoch=9, total_epochs=10
        )

        # 两者不同（动量更新有效）
        self.assertFalse(np.allclose(w_early, w_late))


# ─────────────────────────────────────────────
# 7. 运行入口
# ─────────────────────────────────────────────

def demo():
    """演示 CAGED 管线完整运行"""
    print("=" * 60)
    print("CAGED Demo: 跨境电商长尾商品挖掘")
    print("=" * 60)

    # 构建演示数据
    num_users, num_items = 5, 8
    interactions = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2), (2, 3), (3, 3),
        (0, 5), (1, 6), (2, 7),
    ]

    # 初始化管线
    pipeline = CAGEDRecommendationPipeline(
        num_users=num_users, num_items=num_items,
        embed_dim=8, latent_dim=8, n_layers=2, momentum=0.8
    )
    pipeline.build_graph(interactions)

    # 训练
    print("\n[训练 CAGED 无偏权重估计器...]")
    history = pipeline.train(interactions, epochs=5)
    print(f"  ELBO 变化: {[round(v, 4) for v in history['elbo']]}")

    # 对比推荐结果
    print("\n[对比有偏 vs 无偏推荐（用户 0）]")
    result = pipeline.compare_biased_vs_unbiased(user_id=0, top_k=4)
    print(f"  有偏推荐: {[i for i, _ in result['biased']]}")
    print(f"  无偏推荐: {[i for i, _ in result['unbiased']]}")

    # 流行度分布
    pop = pipeline.caged.item_popularity
    print(f"\n[商品流行度（归一化）]")
    for i in range(num_items):
        bar = "█" * int(pop[i] * 20)
        tag = " ← 爆款" if pop[i] > 0.6 else (" ← 长尾" if pop[i] < 0.2 else "")
        print(f"  商品 {i}: {bar:20s} {pop[i]:.2f}{tag}")

    # NDCG@K 评估
    biased_items = [i for i, _ in result["biased"]]
    relevant_items = [5, 2, 3]  # 假设的真实偏好（含长尾）
    ndcg_biased = ndcg_at_k(biased_items, relevant_items, k=4)
    unbiased_items = [i for i, _ in result["unbiased"]]
    ndcg_unbiased = ndcg_at_k(unbiased_items, relevant_items, k=4)
    print(f"\n[NDCG@4] 有偏: {ndcg_biased:.4f} | 无偏: {ndcg_unbiased:.4f}")

    print("\n[运行单元测试...]")
    return True


if __name__ == "__main__":
    # 先跑 Demo
    demo()
    print("\n" + "=" * 60)
    print("运行单元测试")
    print("=" * 60)
    unittest.main(verbosity=2)
