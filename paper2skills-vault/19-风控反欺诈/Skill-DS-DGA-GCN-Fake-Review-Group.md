---
title: DS-DGA-GCN — 动态图假评论群组检测：冷启动新品防刷评
doc_type: knowledge
module: 19-风控反欺诈
topic: ds-dga-gcn-fake-reviewer-group-detection
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: DS-DGA-GCN — 动态图假评论群组检测

> **领域**: 19-风控反欺诈 | **论文**: Detecting Fake Reviewer Groups in Dynamic Networks (arXiv 2603.08332, 2026-03)

---

## ① 算法原理

**核心思想**：在"产品 → 评论 → 评论者"三方动态异构图上检测刷评团伙群组。不看单条评论文本质量，而是看**评论者之间的网络行为模式**——真实用户构成稀疏随机网络，刷评团伙则共现密集、行为高度同步。

**三方动态图构建**：
- 节点：产品节点 P、评论节点 R、评论者节点 U
- 边：U →（写了）→ R →（针对）→ P，边附时间戳 t
- 动态性：随新评论的到来图结构持续演化

**NFS（Network Feature Scoring）邻居多样性量化**：

$$\text{NFS}(u) = \alpha \cdot D(u) + (1-\alpha) \cdot S(u)$$

其中 D(u) 为邻居多样性（真实用户评论的产品种类多且分散，刷评团伙集中在少数产品），S(u) 为网络自相似性（刷评账号之间的结构相似度高）。真实用户 D 高、S 低；刷评账号 D 低、S 高。

**动态图注意力（时序感知）**：
- 时序权重：近期行为赋予更高权重（防止冷启动期历史不足的稀疏干扰）
- 节点重要性：高 NFS 节点的邻居聚合权重被放大
- 全局结构：跨产品的评论者共现矩阵捕获团伙边界

**为什么冷启动特别难**：新品上线前 72 小时，历史评论极少（稀疏图），传统 GCN 的聚合信息量不足，NFS 通过网络自相似性先验填补稀疏期的特征缺失。

**量化效果**：Amazon 数据集 Accuracy 89.8% / AUROC 0.945；Xiaohongshu 数据集 Accuracy 88.3%。

---

## ② 母婴出海应用案例

### 场景一：母婴新品上架 24 小时内刷评检测

**业务问题**：新款奶瓶（ASIN B0XXXX）上架后 6 小时内突现 38 条 5 星评论，评论者账号注册时间集中在近 7 天、互相之间都购买过同一批产品。肉眼难辨，但网络结构异常明显。

**数据要求**：
| 类型 | 字段 | 来源 |
|------|------|------|
| 评论 | reviewer_id, product_id, rating, timestamp | Amazon API |
| 评论者 | account_age, review_count, verified | 爬取 |
| 产品 | asin, category, launch_date | 内部数据 |

**预期产出**：
- 评论者群组分类：正常 / 可疑 / 虚假团伙
- 可疑群组图谱（哪些账号构成一个团伙）
- NFS 得分热力图（团伙成员的网络指纹）

**业务价值**：新品上线首周是刷评高发期，Amazon 对刷评 Listing 实施降权直至下架。提前 24-48 小时预警，可在被平台检测前主动删除可疑评论并向 Amazon 举报，避免 Listing 下架损失（首周销售额通常 $5,000–$30,000）。

---

### 场景二：WF-D 选品阶段评论质量评估（前置过滤）

**业务问题**：用竞品评论数据做选品分析时（评分分析 Skill），若竞品评论本身大量被刷，分析出来的"用户真实痛点"完全失真，选品决策走偏。

**数据要求**：竞品 ASIN 的公开评论爬取数据（评论者 ID + 时间戳 + 文本）

**预期产出**：
- 评论可信度得分（0~1，按评论者网络质量加权）
- 虚假评论过滤后的干净评论集
- 竞品真实评分估算（去除刷评后的加权平均）

**业务价值**：选品分析的数据质量前置门控。先用 DS-DGA-GCN 清洗虚假评论，再做 [[Skill-AGRS-Aspect-Guided-Review-Summarization]] 真实评分分析，避免选品走偏导致的滞销损失（首批备货通常 10–30 万人民币）。

---

## ③ 代码模板

代码位置：`paper2skills-code/risk_fraud/ds_dga_gcn_group_detection/model.py`

```python
"""
DS-DGA-GCN: 动态图假评论群组检测
论文: Detecting Fake Reviewer Groups in Dynamic Networks (arXiv 2603.08332)
场景: 母婴新品上架刷评检测 + 选品评论质量过滤
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict


# ─── 数据结构 ────────────────────────────────────────────────────────────────

@dataclass
class Review:
    reviewer_id: str
    product_id: str
    rating: float
    timestamp: float       # Unix 时间戳
    verified: bool = False
    review_text: str = ""


@dataclass
class ReviewerNode:
    reviewer_id: str
    account_age_days: int
    total_review_count: int
    reviews: List[Review] = field(default_factory=list)


# ─── 三方动态图 ──────────────────────────────────────────────────────────────

class ReviewerNetworkGraph:
    """产品-评论-评论者 三方动态图（支持时序边添加）"""

    def __init__(self):
        self.reviewer_nodes: Dict[str, ReviewerNode] = {}
        self.product_ids: set = set()
        # 边: reviewer_id -> set of product_ids（co-review 边）
        self.coview_edges: Dict[str, set] = defaultdict(set)
        # 时序边: (reviewer_id, product_id) -> timestamp
        self.temporal_edges: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    def add_review(self, review: Review, reviewer_node: ReviewerNode) -> None:
        """动态添加一条评论（时序边扩展）"""
        rid = review.reviewer_id
        pid = review.product_id

        self.reviewer_nodes[rid] = reviewer_node
        self.product_ids.add(pid)
        self.coview_edges[rid].add(pid)
        self.temporal_edges[(rid, pid)].append(review.timestamp)
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
        # 归一化：超过 10 条/24h 视为高度可疑
        return min(max_burst / 10.0, 1.0)


# ─── NFS 邻居多样性评分 ──────────────────────────────────────────────────────

class NFSScorer:
    """
    Network Feature Scoring:
    D(u) = 邻居多样性（评论产品的分散程度）
    S(u) = 网络自相似性（与其他评论者的结构相似度）
    NFS(u) = α·D(u) + (1-α)·S(u)
    真实用户: D 高, S 低 → NFS 接近 0（正常）
    刷评团伙: D 低, S 高 → NFS 接近 1（异常）
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha  # D 和 S 的权重平衡

    def neighbor_diversity(self, graph: ReviewerNetworkGraph, reviewer_id: str) -> float:
        """
        D(u): 产品多样性得分
        评论的产品越集中 → 多样性越低 → 刷评可能性越高
        使用 Gini 系数的反指标
        """
        products = graph.coview_edges.get(reviewer_id, set())
        if not products:
            return 0.5  # 无数据时返回中性分

        # 每个产品的评论次数
        product_counts = defaultdict(int)
        for (rid, pid), tss in graph.temporal_edges.items():
            if rid == reviewer_id and pid in products:
                product_counts[pid] += len(tss)

        counts = np.array(list(product_counts.values()), dtype=float)
        if counts.sum() == 0:
            return 0.5

        # Gini 系数（越高表示越集中）
        counts = np.sort(counts)
        n = len(counts)
        gini = (2 * np.sum(np.arange(1, n + 1) * counts) / (n * counts.sum()) - (n + 1) / n)
        # D(u) = 1 - Gini（集中度越高，多样性越低，越可疑）
        diversity = 1.0 - float(np.clip(gini, 0, 1))
        return diversity  # 返回多样性：低多样性 → 刷评信号强

    def network_self_similarity(self, graph: ReviewerNetworkGraph, reviewer_id: str) -> float:
        """
        S(u): 与共现评论者的结构相似度
        刷评团伙成员之间的行为高度相似 → S 高
        使用 Jaccard 相似度均值
        """
        co_reviewers = graph.get_co_reviewers(reviewer_id)
        if not co_reviewers:
            return 0.0

        my_products = graph.coview_edges.get(reviewer_id, set())
        similarities = []
        for other_id in co_reviewers[:20]:  # 限制邻居数量避免 O(n²)
            other_products = graph.coview_edges.get(other_id, set())
            intersection = len(my_products & other_products)
            union = len(my_products | other_products)
            jaccard = intersection / union if union > 0 else 0.0
            similarities.append(jaccard)

        return float(np.mean(similarities)) if similarities else 0.0

    def compute_nfs(self, graph: ReviewerNetworkGraph, reviewer_id: str) -> Dict:
        """计算 NFS 综合得分"""
        d_score = self.neighbor_diversity(graph, reviewer_id)
        s_score = self.network_self_similarity(graph, reviewer_id)

        # 刷评团伙特征: D 低(集中) + S 高(相似) → NFS 高
        # 注意: D 是多样性，低多样性意味着高风险，所以用 (1 - d_score)
        fraud_signal_d = 1.0 - d_score  # 转为"集中度"（越高越可疑）
        fraud_signal_s = s_score         # 相似度（越高越可疑）
        nfs = self.alpha * fraud_signal_d + (1 - self.alpha) * fraud_signal_s

        return {
            "reviewer_id": reviewer_id,
            "diversity_score": round(d_score, 4),
            "similarity_score": round(s_score, 4),
            "fraud_signal_d": round(fraud_signal_d, 4),
            "nfs_score": round(nfs, 4),
        }


# ─── 动态图注意力 ────────────────────────────────────────────────────────────

class DynamicGraphAttention:
    """时序感知注意力权重（简化版，无 PyTorch 依赖）"""

    def __init__(self, time_decay: float = 0.01):
        self.time_decay = time_decay  # 时间衰减系数

    def temporal_weight(self, timestamp: float, current_time: float) -> float:
        """
        时间衰减: w(t) = exp(-λ·Δt)
        近期行为权重高，历史行为权重低
        """
        delta_hours = (current_time - timestamp) / 3600
        return float(np.exp(-self.time_decay * delta_hours))

    def compute_attention_weights(
        self,
        graph: ReviewerNetworkGraph,
        reviewer_id: str,
        current_time: float,
        nfs_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        为每个共现评论者计算注意力权重
        权重 = 时序衰减 × NFS 重要性
        """
        co_reviewers = graph.get_co_reviewers(reviewer_id)
        weights = {}
        for cr_id in co_reviewers:
            # 取最近一次共现时间
            shared_products = graph.coview_edges.get(reviewer_id, set()) & \
                              graph.coview_edges.get(cr_id, set())
            last_ts = max(
                (ts for pid in shared_products
                 for ts in graph.temporal_edges.get((cr_id, pid), [])),
                default=current_time - 86400,
            )
            t_weight = self.temporal_weight(last_ts, current_time)
            nfs = nfs_scores.get(cr_id, 0.0)
            weights[cr_id] = t_weight * (1 + nfs)  # 高 NFS 节点贡献更大

        # Softmax 归一化
        if weights:
            vals = np.array(list(weights.values()))
            vals = np.exp(vals - vals.max())
            vals /= vals.sum()
            weights = {k: float(v) for k, v in zip(weights.keys(), vals)}

        return weights


# ─── 群组分类器 ──────────────────────────────────────────────────────────────

class DSDGAGCNDetector:
    """
    DS-DGA-GCN 假评论群组分类器
    标签：NORMAL(正常) / SUSPICIOUS(可疑) / FAKE_GROUP(虚假团伙)
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

    def ingest_reviews(self, reviews: List[Review], reviewer_nodes: Dict[str, ReviewerNode]) -> None:
        """批量导入评论流（按时间戳顺序）"""
        reviews_sorted = sorted(reviews, key=lambda r: r.timestamp)
        for review in reviews_sorted:
            node = reviewer_nodes.get(review.reviewer_id)
            if node:
                self.graph.add_review(review, node)

    def compute_all_nfs(self) -> Dict[str, float]:
        """计算所有评论者的 NFS 得分"""
        for rid in self.graph.reviewer_nodes:
            result = self.nfs_scorer.compute_nfs(self.graph, rid)
            self._nfs_cache[rid] = result["nfs_score"]
        return self._nfs_cache

    def classify_reviewer(
        self,
        reviewer_id: str,
        current_time: float,
    ) -> Dict:
        """对单个评论者进行群组分类"""
        nfs = self._nfs_cache.get(reviewer_id)
        if nfs is None:
            result = self.nfs_scorer.compute_nfs(self.graph, reviewer_id)
            nfs = result["nfs_score"]
            self._nfs_cache[reviewer_id] = nfs

        # 获取注意力加权的群组上下文
        attn_weights = self.attention.compute_attention_weights(
            self.graph, reviewer_id, current_time, self._nfs_cache
        )
        # 群组平均 NFS（邻居的加权 NFS）
        group_nfs = sum(
            w * self._nfs_cache.get(cr_id, 0.0)
            for cr_id, w in attn_weights.items()
        ) if attn_weights else 0.0

        # 综合得分（个人 NFS + 群组 NFS 加权）
        combined_score = 0.7 * nfs + 0.3 * group_nfs

        if combined_score >= self.nfs_thresh_fake:
            label = self.FAKE_GROUP
        elif combined_score >= self.nfs_thresh_sus:
            label = self.SUSPICIOUS
        else:
            label = self.NORMAL

        return {
            "reviewer_id": reviewer_id,
            "individual_nfs": round(nfs, 4),
            "group_nfs": round(group_nfs, 4),
            "combined_score": round(combined_score, 4),
            "label": label,
            "co_reviewer_count": len(attn_weights),
        }

    def classify_product_reviews(self, product_id: str, current_time: float) -> Dict:
        """对某产品所有评论者进行群组分类，输出检测报告"""
        # 找出该产品的所有评论者
        reviewers = [
            rid for rid, prods in self.graph.coview_edges.items()
            if product_id in prods
        ]

        if not reviewers:
            return {"product_id": product_id, "reviewers": [], "summary": {}}

        self.compute_all_nfs()
        results = [self.classify_reviewer(rid, current_time) for rid in reviewers]

        burst_score = self.graph.get_review_burst_score(product_id)
        counts = {self.NORMAL: 0, self.SUSPICIOUS: 0, self.FAKE_GROUP: 0}
        for r in results:
            counts[r["label"]] += 1

        fake_ratio = counts[self.FAKE_GROUP] / len(results)
        alert_level = (
            "HIGH" if fake_ratio > 0.3 or burst_score > self.burst_thresh
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


# ─── 测试：模拟母婴产品上架后 72 小时的评论流 ────────────────────────────────

def test_baby_product_launch():
    """
    场景：奶瓶新品（ASIN B0TEST001）上架后 72 小时
    注入 30 条正常评论 + 15 条刷评团伙评论
    验证: 刷评团伙被正确识别
    """
    import time

    base_time = 1748736000.0  # 2026-06-01 00:00:00 UTC
    product_id = "B0TEST001"

    reviews: List[Review] = []
    reviewer_nodes: Dict[str, ReviewerNode] = {}

    # ── 正常评论者（30人，各自评论不同产品，时间分散）
    for i in range(30):
        rid = f"normal_user_{i:03d}"
        # 正常用户评论多种不同产品（历史丰富）
        other_products = [f"B0PROD{j:03d}" for j in range(i % 5, i % 5 + 8)]
        reviewer_nodes[rid] = ReviewerNode(
            reviewer_id=rid,
            account_age_days=np.random.randint(180, 1800),
            total_review_count=np.random.randint(10, 100),
        )
        # 历史评论（分散在不同产品）
        for prod in other_products[:3]:
            r = Review(
                reviewer_id=rid,
                product_id=prod,
                rating=np.random.choice([3.0, 4.0, 5.0]),
                timestamp=base_time - np.random.uniform(3600, 72 * 3600),
                verified=True,
            )
            reviews.append(r)
        # 本次对新品的评论（时间分散，不集中）
        reviews.append(Review(
            reviewer_id=rid,
            product_id=product_id,
            rating=np.random.choice([3.0, 4.0, 5.0]),
            timestamp=base_time + np.random.uniform(0, 72 * 3600),
            verified=True,
        ))

    # ── 刷评团伙（15人，集中在 6 小时内，只评论 1-2 个产品）
    fraud_base_time = base_time + 2 * 3600  # 上架 2 小时后集中刷评
    for i in range(15):
        rid = f"fraud_bot_{i:03d}"
        reviewer_nodes[rid] = ReviewerNode(
            reviewer_id=rid,
            account_age_days=np.random.randint(1, 15),  # 新账号
            total_review_count=np.random.randint(1, 5),
        )
        # 团伙成员之间共同评论了同一个"刷单产品"
        shared_product = "B0SHARED001"
        reviews.append(Review(
            reviewer_id=rid,
            product_id=shared_product,
            rating=5.0,
            timestamp=fraud_base_time - 3600,  # 在刷评前 1 小时都去了同一产品
        ))
        # 集中时间段刷评新品
        reviews.append(Review(
            reviewer_id=rid,
            product_id=product_id,
            rating=5.0,
            timestamp=fraud_base_time + np.random.uniform(0, 3600),  # 1 小时内集中
            verified=False,
        ))

    # ── 运行检测
    detector = DSDGAGCNDetector(
        nfs_threshold_suspicious=0.35,
        nfs_threshold_fake=0.60,
        burst_threshold=0.5,
    )
    detector.ingest_reviews(reviews, reviewer_nodes)
    report = detector.classify_product_reviews(product_id, base_time + 72 * 3600)

    print("=" * 60)
    print(f"[DS-DGA-GCN] 产品 {product_id} 评论群组检测报告")
    print("=" * 60)
    print(f"总评论者数: {report['total_reviewers']}")
    print(f"24h 爆发得分: {report['burst_score_24h']:.4f} (>0.5 为高风险)")
    print(f"标签分布: {report['label_distribution']}")
    print(f"虚假评论比例: {report['fake_ratio']:.1%}")
    print(f"告警级别: {report['alert_level']}")
    print()

    # 打印刷评团伙成员
    fake_reviewers = [r for r in report["reviewers"] if r["label"] == "FAKE_GROUP"]
    print(f"[虚假团伙成员] 识别到 {len(fake_reviewers)} 人:")
    for fr in fake_reviewers[:5]:
        print(f"  {fr['reviewer_id']}: NFS={fr['combined_score']:.3f}")

    # 断言验证
    assert report["alert_level"] in ["MEDIUM", "HIGH"], \
        f"期望 MEDIUM/HIGH 告警，实际: {report['alert_level']}"
    fake_count = report["label_distribution"].get("FAKE_GROUP", 0)
    assert fake_count >= 8, \
        f"期望检测到至少 8 个虚假团伙成员，实际: {fake_count}"
    print()
    print("[✓] DS-DGA-GCN 母婴新品刷评检测测试通过")
    return report


if __name__ == "__main__":
    test_baby_product_launch()
```

---

## ④ 技能关联

- **前置**：[[Skill-Review-Fraud-Detection]] / [[Skill-FraudSquad-LLM-Review-Detection]] / [[Skill-HGT-Heterogeneous-Graph-Transformer]]
- **延伸**：[[Skill-Transaction-Anomaly-Detection]] / [[Skill-Click-Fraud-Detection]]
- **可组合**：[[Skill-AGRS-Aspect-Guided-Review-Summarization]]（先过滤虚假评论，再做真实摘要）/ [[Skill-Flowr-Supply-Chain-MAS]] / [[Skill-Category-Compliance-Prescan]]

---
- **关联**：[[Skill-AnchorCrafter-Virtual-Anchor-Demo]]
- **相关**：[[Skill-Customer-Churn-Prediction]]

## ⑤ 商业价值

| 指标 | 评估 | 说明 |
|------|------|------|
| ROI | 高 | 新品首周销售额 $5,000–$30,000，Listing 被下架损失 100% |
| 实施难度 | ⭐⭐⭐☆☆ | 需图数据库或动态图处理，无需 GPU |
| 优先级 | ⭐⭐⭐⭐⭐ | 新品上线是刷评高发期，且平台处罚成本极高 |

**量化依据**：Amazon 数据集 Accuracy 89.8% / AUROC 0.945；Xiaohongshu 88.3%。  
新品首周是刷评高发期，早发现早处理可避免被 Amazon 下架；DS-DGA-GCN 专门针对冷启动（数据稀疏）场景优化，比通用 GCN 在新品前 24 小时准确率高约 12%。

**参考论文**：arXiv 2603.08332，2026 年 3 月
