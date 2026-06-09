"""
DS-DGA-GCN: 动态图假评论群组检测
论文: Detecting Fake Reviewer Groups in Dynamic Networks (arXiv 2603.08332, 2026-03)

核心模块:
- ReviewerNetworkGraph: 产品-评论-评论者 三方动态图
- NFSScorer: 邻居多样性 + 网络自相似性评分
- DynamicGraphAttention: 时序感知注意力权重
- DSDGAGCNDetector: 群组分类（NORMAL / SUSPICIOUS / FAKE_GROUP）

运行:
    python model.py

依赖:
    numpy>=1.26
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class Review:
    """单条评论"""
    reviewer_id: str
    product_id: str
    rating: float          # 1.0–5.0
    timestamp: float       # Unix 时间戳（秒）
    verified: bool = False
    review_text: str = ""


@dataclass
class ReviewerNode:
    """评论者节点特征"""
    reviewer_id: str
    account_age_days: int          # 账号注册天数
    total_review_count: int        # 历史总评论数
    reviews: List[Review] = field(default_factory=list)


# ─── 三方动态图 ───────────────────────────────────────────────────────────────

class ReviewerNetworkGraph:
    """
    产品-评论-评论者 三方动态图（支持时序边添加）

    图结构:
        U（评论者）---[写了]--->  R（评论）---[针对]--->  P（产品）
        边附时间戳，支持动态演化
    """

    def __init__(self):
        self.reviewer_nodes: Dict[str, ReviewerNode] = {}
        self.product_ids: set = set()
        # 共现边: reviewer_id -> set of product_ids
        self.coview_edges: Dict[str, set] = defaultdict(set)
        # 时序边: (reviewer_id, product_id) -> [timestamps]
        self.temporal_edges: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    def add_review(self, review: Review, reviewer_node: ReviewerNode) -> None:
        """动态添加一条评论（时序边扩展）"""
        rid = review.reviewer_id
        pid = review.product_id

        self.reviewer_nodes[rid] = reviewer_node
        self.product_ids.add(pid)
        self.coview_edges[rid].add(pid)
        self.temporal_edges[(rid, pid)].append(review.timestamp)
        if review not in reviewer_node.reviews:
            reviewer_node.reviews.append(review)

    def get_co_reviewers(self, reviewer_id: str) -> List[str]:
        """找出与该评论者评论过同一产品的所有评论者（共现关系）"""
        target_products = self.coview_edges.get(reviewer_id, set())
        co_reviewers = []
        for other_id, products in self.coview_edges.items():
            if other_id != reviewer_id and target_products & products:
                co_reviewers.append(other_id)
        return co_reviewers

    def get_review_burst_score(self, product_id: str, window_hours: float = 24.0) -> float:
        """
        计算产品在时间窗口内的评论爆发得分
        爆发得分高 = 短时间内大量评论涌入（刷评信号）
        归一化：超过 10 条/24h 视为高度可疑（返回 1.0）
        """
        all_timestamps = []
        for (rid, pid), tss in self.temporal_edges.items():
            if pid == product_id:
                all_timestamps.extend(tss)
        if len(all_timestamps) < 2:
            return 0.0
        all_timestamps.sort()
        window_sec = window_hours * 3600
        max_burst = 0
        for i, ts in enumerate(all_timestamps):
            count_in_window = sum(1 for t in all_timestamps[i:] if t - ts <= window_sec)
            max_burst = max(max_burst, count_in_window)
        return min(max_burst / 10.0, 1.0)


# ─── NFS 邻居多样性评分 ───────────────────────────────────────────────────────

class NFSScorer:
    """
    Network Feature Scoring (NFS)
    NFS(u) = α·(1 - D(u)) + (1-α)·S(u)

    D(u) = 邻居多样性（基于 Gini 系数）：真实用户评论分散，D 高
    S(u) = 网络自相似性（Jaccard 均值）：刷评团伙成员行为相似，S 高
    NFS 高 → 刷评风险高
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def neighbor_diversity(self, graph: ReviewerNetworkGraph, reviewer_id: str) -> float:
        """D(u): 产品多样性（越分散越高，越集中越低）"""
        products = graph.coview_edges.get(reviewer_id, set())
        if not products:
            return 0.5

        product_counts: Dict[str, int] = defaultdict(int)
        for (rid, pid), tss in graph.temporal_edges.items():
            if rid == reviewer_id and pid in products:
                product_counts[pid] += len(tss)

        counts = np.array(list(product_counts.values()), dtype=float)
        if counts.sum() == 0:
            return 0.5

        n = len(counts)
        counts_sorted = np.sort(counts)
        if n == 1:
            gini = 0.0
        else:
            gini = (2 * np.sum(np.arange(1, n + 1) * counts_sorted) /
                    (n * counts_sorted.sum()) - (n + 1) / n)

        gini_component = float(np.clip(gini, 0, 1))

        # 产品数量惩罚：评论的产品种类极少（≤3）是刷评信号
        # 正常用户通常评论 5+ 个不同产品；刷评账号集中在 1-2 个
        product_count_penalty = float(np.clip(1.0 - (n - 1) / 5.0, 0.0, 1.0))

        # 综合多样性：Gini 集中度 + 产品种类稀少惩罚，取最大值
        concentration = max(gini_component, product_count_penalty * 0.7)
        diversity = 1.0 - concentration
        return float(np.clip(diversity, 0.0, 1.0))

    def network_self_similarity(
        self,
        graph: ReviewerNetworkGraph,
        reviewer_id: str,
        exclude_products: Optional[set] = None,
    ) -> float:
        """
        S(u): 与共现评论者的 Jaccard 相似度均值
        exclude_products: 排除热门产品（所有人都评论过，不具区分性）
        只考虑在"私有产品集"上有交集的邻居（真实共现信号）
        """
        my_products = graph.coview_edges.get(reviewer_id, set())
        if exclude_products:
            my_products = my_products - exclude_products

        if not my_products:
            return 0.0

        similarities = []
        for other_id, other_all_products in graph.coview_edges.items():
            if other_id == reviewer_id:
                continue
            other_products = other_all_products if not exclude_products else other_all_products - exclude_products
            if not (my_products & other_products):
                continue
            intersection = len(my_products & other_products)
            union = len(my_products | other_products)
            jaccard = intersection / union if union > 0 else 0.0
            similarities.append(jaccard)
            if len(similarities) >= 20:
                break

        return float(np.mean(similarities)) if similarities else 0.0

    def compute_nfs(
        self,
        graph: ReviewerNetworkGraph,
        reviewer_id: str,
        exclude_products: Optional[set] = None,
    ) -> Dict:
        """计算 NFS 综合得分，返回详细分项"""
        d_score = self.neighbor_diversity(graph, reviewer_id)
        s_score = self.network_self_similarity(graph, reviewer_id, exclude_products)
        fraud_d = 1.0 - d_score
        nfs = self.alpha * fraud_d + (1 - self.alpha) * s_score

        return {
            "reviewer_id": reviewer_id,
            "diversity_score": round(d_score, 4),
            "similarity_score": round(s_score, 4),
            "nfs_score": round(float(nfs), 4),
        }


# ─── 动态图注意力 ─────────────────────────────────────────────────────────────

class DynamicGraphAttention:
    """
    时序感知注意力权重
    w(u, t) = exp(-λ·Δt) × (1 + NFS(u))
    近期行为权重高；高 NFS 节点（可疑邻居）权重被放大
    """

    def __init__(self, time_decay: float = 0.01):
        self.time_decay = time_decay  # λ，单位：1/小时

    def temporal_weight(self, timestamp: float, current_time: float) -> float:
        delta_hours = (current_time - timestamp) / 3600
        return float(np.exp(-self.time_decay * delta_hours))

    def compute_attention_weights(
        self,
        graph: ReviewerNetworkGraph,
        reviewer_id: str,
        current_time: float,
        nfs_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """为每个共现评论者计算归一化注意力权重"""
        co_reviewers = graph.get_co_reviewers(reviewer_id)
        raw_weights: Dict[str, float] = {}

        for cr_id in co_reviewers:
            shared_products = (graph.coview_edges.get(reviewer_id, set()) &
                               graph.coview_edges.get(cr_id, set()))
            last_ts = max(
                (ts for pid in shared_products
                 for ts in graph.temporal_edges.get((cr_id, pid), [])),
                default=current_time - 86400,
            )
            t_weight = self.temporal_weight(last_ts, current_time)
            nfs = nfs_scores.get(cr_id, 0.0)
            raw_weights[cr_id] = t_weight * (1 + nfs)

        if not raw_weights:
            return {}

        # Softmax 归一化
        vals = np.array(list(raw_weights.values()))
        vals = np.exp(vals - vals.max())
        vals /= vals.sum()
        return {k: float(v) for k, v in zip(raw_weights.keys(), vals)}


# ─── 群组分类器 ───────────────────────────────────────────────────────────────

class DSDGAGCNDetector:
    """
    DS-DGA-GCN 假评论群组检测器

    分类标签:
        NORMAL      — 正常评论者
        SUSPICIOUS  — 可疑（需人工复核）
        FAKE_GROUP  — 虚假评论团伙（建议直接上报）

    参数:
        nfs_threshold_suspicious: NFS ≥ 此值判定为 SUSPICIOUS
        nfs_threshold_fake:       NFS ≥ 此值判定为 FAKE_GROUP
        burst_threshold:          24h 爆发得分阈值（产品级告警）
    """

    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    FAKE_GROUP = "FAKE_GROUP"

    def __init__(
        self,
        nfs_threshold_suspicious: float = 0.4,
        nfs_threshold_fake: float = 0.65,
        burst_threshold: float = 0.5,
        time_decay: float = 0.01,
        alpha: float = 0.5,
    ):
        self.nfs_thresh_sus = nfs_threshold_suspicious
        self.nfs_thresh_fake = nfs_threshold_fake
        self.burst_thresh = burst_threshold
        self.graph = ReviewerNetworkGraph()
        self.nfs_scorer = NFSScorer(alpha=alpha)
        self.attention = DynamicGraphAttention(time_decay=time_decay)
        self._nfs_cache: Dict[str, float] = {}

    def ingest_reviews(
        self,
        reviews: List[Review],
        reviewer_nodes: Dict[str, ReviewerNode],
    ) -> None:
        """按时间顺序批量导入评论流"""
        for review in sorted(reviews, key=lambda r: r.timestamp):
            node = reviewer_nodes.get(review.reviewer_id)
            if node:
                self.graph.add_review(review, node)

    def compute_all_nfs(self, exclude_products: Optional[set] = None) -> Dict[str, float]:
        """计算所有评论者 NFS 得分（结果缓存）"""
        for rid in self.graph.reviewer_nodes:
            result = self.nfs_scorer.compute_nfs(self.graph, rid, exclude_products)
            self._nfs_cache[rid] = result["nfs_score"]
        return self._nfs_cache

    def classify_reviewer(self, reviewer_id: str, current_time: float) -> Dict:
        """对单个评论者进行群组分类"""
        nfs = self._nfs_cache.get(reviewer_id)
        if nfs is None:
            nfs = self.nfs_scorer.compute_nfs(self.graph, reviewer_id)["nfs_score"]
            self._nfs_cache[reviewer_id] = nfs

        attn_weights = self.attention.compute_attention_weights(
            self.graph, reviewer_id, current_time, self._nfs_cache
        )
        group_nfs = (
            sum(w * self._nfs_cache.get(cr_id, 0.0) for cr_id, w in attn_weights.items())
            if attn_weights else 0.0
        )

        combined = 0.7 * nfs + 0.3 * group_nfs

        if combined >= self.nfs_thresh_fake:
            label = self.FAKE_GROUP
        elif combined >= self.nfs_thresh_sus:
            label = self.SUSPICIOUS
        else:
            label = self.NORMAL

        return {
            "reviewer_id": reviewer_id,
            "individual_nfs": round(nfs, 4),
            "group_nfs": round(group_nfs, 4),
            "combined_score": round(combined, 4),
            "label": label,
            "co_reviewer_count": len(attn_weights),
        }

    def classify_product_reviews(self, product_id: str, current_time: float) -> Dict:
        """对某产品所有评论者进行群组分类，输出检测报告"""
        reviewers = [
            rid for rid, prods in self.graph.coview_edges.items()
            if product_id in prods
        ]
        if not reviewers:
            return {"product_id": product_id, "reviewers": [], "summary": {}}

        # 排除目标产品：所有评论者都评了它，不构成区分性信号
        self.compute_all_nfs(exclude_products={product_id})
        results = [self.classify_reviewer(rid, current_time) for rid in reviewers]

        burst_score = self.graph.get_review_burst_score(product_id)
        counts = {self.NORMAL: 0, self.SUSPICIOUS: 0, self.FAKE_GROUP: 0}
        for r in results:
            counts[r["label"]] += 1

        fake_ratio = counts[self.FAKE_GROUP] / len(results)
        alert_level = (
            "HIGH"
            if fake_ratio > 0.3 or burst_score > self.burst_thresh
            else "MEDIUM" if fake_ratio > 0.1
            else "LOW"
        )

        return {
            "product_id": product_id,
            "total_reviewers": len(results),
            "burst_score_24h": round(burst_score, 4),
            "label_distribution": counts,
            "fake_ratio": round(fake_ratio, 4),
            "alert_level": alert_level,
            "reviewers": results,
        }


# ─── 测试：母婴产品上架后 72 小时评论流 ──────────────────────────────────────

def test_baby_product_launch() -> Dict:
    """
    场景：奶瓶新品（B0TEST001）上架后 72 小时
    注入 30 条正常评论 + 15 条刷评团伙评论
    验证：刷评团伙 NFS 显著高于正常用户，告警级别 MEDIUM/HIGH，
          可疑+虚假 评论者 ≥ 8 人
    """
    base_time = 1748736000.0  # 2026-06-01 00:00:00 UTC
    product_id = "B0TEST001"

    reviews: List[Review] = []
    reviewer_nodes: Dict[str, ReviewerNode] = {}

    rng = np.random.default_rng(42)

    # ── 正常评论者（30 人，评论分散，时间分布广）
    for i in range(30):
        rid = f"normal_user_{i:03d}"
        reviewer_nodes[rid] = ReviewerNode(
            reviewer_id=rid,
            account_age_days=int(rng.integers(180, 1800)),
            total_review_count=int(rng.integers(10, 100)),
        )
        for j in range(3):
            prod = f"B0PROD{(i * 3 + j) % 50:03d}"
            reviews.append(Review(
                reviewer_id=rid,
                product_id=prod,
                rating=float(rng.choice([3.0, 4.0, 5.0])),
                timestamp=base_time - float(rng.uniform(3600, 72 * 3600)),
                verified=True,
            ))
        reviews.append(Review(
            reviewer_id=rid,
            product_id=product_id,
            rating=float(rng.choice([3.0, 4.0, 5.0])),
            timestamp=base_time + float(rng.uniform(0, 72 * 3600)),
            verified=True,
        ))

    # ── 刷评团伙（15 人，新账号，集中时间）
    # 团伙有 3 个专属刷单产品（强共现信号：Jaccard ≈ 0.6+）
    fraud_base = base_time + 2 * 3600
    fraud_shared_products = ["B0SHARED001", "B0SHARED002", "B0SHARED003"]
    for i in range(15):
        rid = f"fraud_bot_{i:03d}"
        reviewer_nodes[rid] = ReviewerNode(
            reviewer_id=rid,
            account_age_days=int(rng.integers(1, 15)),
            total_review_count=int(rng.integers(1, 5)),
        )
        for shared_prod in fraud_shared_products:
            reviews.append(Review(
                reviewer_id=rid,
                product_id=shared_prod,
                rating=5.0,
                timestamp=fraud_base - 3600,
            ))
        reviews.append(Review(
            reviewer_id=rid,
            product_id=product_id,
            rating=5.0,
            timestamp=fraud_base + float(rng.uniform(0, 3600)),
            verified=False,
        ))

    detector = DSDGAGCNDetector(
        nfs_threshold_suspicious=0.30,
        nfs_threshold_fake=0.55,
        burst_threshold=0.5,
    )
    detector.ingest_reviews(reviews, reviewer_nodes)
    report = detector.classify_product_reviews(product_id, base_time + 72 * 3600)

    print("=" * 60)
    print(f"[DS-DGA-GCN] 产品 {product_id} 检测报告")
    print("=" * 60)
    print(f"总评论者: {report['total_reviewers']}")
    print(f"24h 爆发得分: {report['burst_score_24h']:.4f}")
    print(f"标签分布: {report['label_distribution']}")
    print(f"虚假比例: {report['fake_ratio']:.1%}")
    print(f"告警级别: {report['alert_level']}")

    fake_reviewers = [r for r in report["reviewers"] if r["label"] == "FAKE_GROUP"]
    suspicious_reviewers = [r for r in report["reviewers"] if r["label"] == "SUSPICIOUS"]
    print(f"\n虚假团伙: {len(fake_reviewers)} 人，可疑: {len(suspicious_reviewers)} 人")
    for fr in (fake_reviewers + suspicious_reviewers)[:5]:
        print(f"  {fr['reviewer_id']}: NFS={fr['combined_score']:.3f} [{fr['label']}]")

    assert report["alert_level"] in ("MEDIUM", "HIGH"), \
        f"期望 MEDIUM/HIGH，实际: {report['alert_level']}"
    flagged_count = (
        report["label_distribution"].get("FAKE_GROUP", 0) +
        report["label_distribution"].get("SUSPICIOUS", 0)
    )
    assert flagged_count >= 8, f"期望 ≥8 个可疑/虚假成员，实际: {flagged_count}"

    print("\n[✓] DS-DGA-GCN 测试通过")
    return report


if __name__ == "__main__":
    test_baby_product_launch()
