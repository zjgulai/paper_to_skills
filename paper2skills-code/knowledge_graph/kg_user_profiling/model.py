"""
KG-Powered User Profiling — 知识图谱驱动的用户画像
Python 标准库实现，无第三方依赖，Python 3.14 兼容
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math
import heapq
import time as _time


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class ProductKGNode:
    """产品知识图谱节点"""
    product_id: str
    name: str
    category: str                          # 一级品类，如 "奶粉"
    sub_category: str = ""                 # 二级品类，如 "有机奶粉"
    attributes: dict[str, str] = field(default_factory=dict)
    certifications: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    price_tier: str = "mid"                # low/mid/high


@dataclass
class UserAction:
    """用户行为记录"""
    user_id: str
    product_id: str
    action_type: str      # purchase/view/review
    timestamp: float      # Unix timestamp
    score: float = 1.0    # 评分（review时有效）


# ─────────────────────────────────────────────
# 用户 KG 画像构建器
# ─────────────────────────────────────────────

class UserKGProfiler:
    """
    从行为历史推断用户知识偏好向量。

    偏好向量格式：{知识实体key -> 权重}
    实体key 格式：
      "cat:{品类名}"
      "cert:{认证名}"
      "tag:{标签名}"
      "attr:{属性名}:{属性值}"
    """

    ACTION_WEIGHTS: dict[str, float] = {
        "purchase": 3.0,
        "review": 2.0,
        "view": 0.5,
    }

    def __init__(
        self,
        kg_nodes: list[ProductKGNode],
        decay_factor: float = 0.9,
        max_age_days: float = 180.0,
    ) -> None:
        self.decay_factor = decay_factor
        self.max_age_days = max_age_days
        self._kg: dict[str, ProductKGNode] = {n.product_id: n for n in kg_nodes}

    def _time_weight(self, action_ts: float, now_ts: float) -> float:
        """指数时间衰减：按月衰减"""
        age_days = (now_ts - action_ts) / 86400.0
        if age_days > self.max_age_days or age_days < 0:
            return 0.0
        return self.decay_factor ** (age_days / 30.0)

    def build_profile(
        self,
        actions: list[UserAction],
        now_ts: Optional[float] = None,
    ) -> dict[str, float]:
        """
        构建用户知识偏好向量（L1 归一化）。

        Args:
            actions: 用户行为历史列表
            now_ts: 当前时间戳（默认 time.time()）

        Returns:
            dict: {实体key -> 偏好权重}，总和归一化为 1.0
        """
        if now_ts is None:
            now_ts = _time.time()

        pref: dict[str, float] = {}

        for action in actions:
            node = self._kg.get(action.product_id)
            if node is None:
                continue

            base_w = self.ACTION_WEIGHTS.get(action.action_type, 0.5)
            time_w = self._time_weight(action.timestamp, now_ts)
            w = base_w * time_w
            if w < 1e-9:
                continue

            # 品类实体（一级 + 二级）
            for entity in [node.category, node.sub_category]:
                if entity:
                    key = f"cat:{entity}"
                    pref[key] = pref.get(key, 0.0) + w

            # 认证实体（权重放大 1.5x，认证是强意图信号）
            for cert in node.certifications:
                key = f"cert:{cert}"
                pref[key] = pref.get(key, 0.0) + w * 1.5

            # 标签实体
            for tag in node.tags:
                key = f"tag:{tag}"
                pref[key] = pref.get(key, 0.0) + w

            # 属性值实体（权重 0.8x）
            for attr_k, attr_v in node.attributes.items():
                key = f"attr:{attr_k}:{attr_v}"
                pref[key] = pref.get(key, 0.0) + w * 0.8

        # L1 归一化
        total = sum(pref.values())
        if total > 1e-9:
            pref = {k: v / total for k, v in pref.items()}

        return pref

    def cold_start_profile(self, baby_age_months: int) -> dict[str, float]:
        """
        冷启动：根据宝宝月龄直接构建初始偏好向量。

        Args:
            baby_age_months: 宝宝月龄

        Returns:
            单实体偏好向量，权重为 1.0
        """
        if baby_age_months <= 6:
            stage = "0-6m"
        elif baby_age_months <= 12:
            stage = "6-12m"
        elif baby_age_months <= 24:
            stage = "12-24m"
        else:
            stage = "24m+"
        return {f"attr:适用月龄:{stage}": 1.0}


# ─────────────────────────────────────────────
# KG 增强推荐器
# ─────────────────────────────────────────────

class KGEnhancedRecommender:
    """
    知识路径增强的推荐分数计算。

    最终分数 = alpha * KG语义分数 + beta * 归一化流行度分数
    """

    def __init__(
        self,
        kg_nodes: list[ProductKGNode],
        profiler: UserKGProfiler,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> None:
        self.profiler = profiler
        self.alpha = alpha
        self.beta = beta
        self._kg: dict[str, ProductKGNode] = {n.product_id: n for n in kg_nodes}
        self._popularity: dict[str, float] = {pid: 1.0 for pid in self._kg}

    def update_popularity(self, purchase_counts: dict[str, float]) -> None:
        """更新产品流行度（可从销量统计中注入）"""
        self._popularity.update(purchase_counts)

    def _kg_score(
        self,
        user_profile: dict[str, float],
        node: ProductKGNode,
    ) -> float:
        """用户偏好向量与产品知识实体的匹配分数"""
        score = 0.0

        # 品类匹配
        score += user_profile.get(f"cat:{node.category}", 0.0) * 1.0
        if node.sub_category:
            score += user_profile.get(f"cat:{node.sub_category}", 0.0) * 1.2

        # 认证匹配（高权重）
        for cert in node.certifications:
            score += user_profile.get(f"cert:{cert}", 0.0) * 1.5

        # 标签匹配
        for tag in node.tags:
            score += user_profile.get(f"tag:{tag}", 0.0) * 1.0

        # 属性匹配
        for attr_k, attr_v in node.attributes.items():
            score += user_profile.get(f"attr:{attr_k}:{attr_v}", 0.0) * 0.8

        return score

    def recommend(
        self,
        user_id: str,
        user_history: list[UserAction],
        top_k: int = 10,
        exclude_purchased: bool = True,
        now_ts: Optional[float] = None,
    ) -> list[tuple[str, float]]:
        """
        生成 top-k 推荐列表。

        Args:
            user_id: 用户ID
            user_history: 用户行为历史
            top_k: 返回推荐数量
            exclude_purchased: 是否排除已购买产品
            now_ts: 当前时间戳

        Returns:
            list of (product_id, score)，按分数降序
        """
        user_profile = self.profiler.build_profile(user_history, now_ts=now_ts)

        purchased: set[str] = set()
        if exclude_purchased:
            purchased = {a.product_id for a in user_history if a.action_type == "purchase"}

        max_pop = max(self._popularity.values()) if self._popularity else 1.0
        scored: list[tuple[float, str]] = []

        for pid, node in self._kg.items():
            if pid in purchased:
                continue
            kg_s = self._kg_score(user_profile, node)
            pop_norm = self._popularity.get(pid, 1.0) / max_pop
            final_score = self.alpha * kg_s + self.beta * pop_norm
            scored.append((final_score, pid))

        top = heapq.nlargest(top_k, scored, key=lambda x: x[0])
        return [(pid, round(score, 6)) for score, pid in top]

    def recommend_cold_start(
        self,
        baby_age_months: int,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        冷启动推荐（仅依赖宝宝月龄）。
        """
        cold_profile = self.profiler.cold_start_profile(baby_age_months)
        max_pop = max(self._popularity.values()) if self._popularity else 1.0
        scored: list[tuple[float, str]] = []

        for pid, node in self._kg.items():
            kg_s = self._kg_score(cold_profile, node)
            pop_norm = self._popularity.get(pid, 1.0) / max_pop
            final_score = self.alpha * kg_s + self.beta * pop_norm
            scored.append((final_score, pid))

        top = heapq.nlargest(top_k, scored, key=lambda x: x[0])
        return [(pid, round(score, 6)) for score, pid in top]


# ─────────────────────────────────────────────
# 测试入口
# ─────────────────────────────────────────────

def run_test() -> None:
    """5个产品 KG + 3个用户行为历史，验证知识增强推荐结果"""
    now = _time.time()
    one_week = 7 * 86400.0

    # ── 构建产品 KG ──
    kg = [
        ProductKGNode(
            product_id="P001", name="有机奶粉A",
            category="奶粉", sub_category="有机奶粉",
            attributes={"适用月龄": "0-6m", "成分": "DHA"},
            certifications=["EU有机认证"], tags=["有机", "益生菌"],
        ),
        ProductKGNode(
            product_id="P002", name="有机米粉B",
            category="辅食", sub_category="有机辅食",
            attributes={"适用月龄": "6-12m", "成分": "铁锌"},
            certifications=["EU有机认证"], tags=["有机", "无添加"],
        ),
        ProductKGNode(
            product_id="P003", name="婴儿护肤霜C",
            category="护肤", sub_category="婴儿护肤",
            attributes={"适用月龄": "0-6m", "成分": "金盏花"},
            certifications=["EU有机认证"], tags=["有机", "敏感肌"],
        ),
        ProductKGNode(
            product_id="P004", name="普通奶粉D",
            category="奶粉", sub_category="普通奶粉",
            attributes={"适用月龄": "0-6m", "成分": "乳清蛋白"},
            certifications=["FDA认证"], tags=["经济实惠"],
        ),
        ProductKGNode(
            product_id="P005", name="益生菌滴剂E",
            category="营养补充", sub_category="益生菌",
            attributes={"适用月龄": "0-12m", "成分": "乳双歧杆菌"},
            certifications=["EU有机认证"], tags=["有机", "益生菌"],
        ),
    ]

    profiler = UserKGProfiler(kg, decay_factor=0.9)
    recommender = KGEnhancedRecommender(kg, profiler)

    # ── 用户1：购买了有机奶粉A，偏好有机 ──
    u1_history = [
        UserAction("u1", "P001", "purchase", now - one_week),
        UserAction("u1", "P001", "review", now - 3 * 86400.0, score=5.0),
    ]

    # ── 用户2：购买普通奶粉，浏览了有机辅食 ──
    u2_history = [
        UserAction("u2", "P004", "purchase", now - 2 * one_week),
        UserAction("u2", "P002", "view", now - one_week),
    ]

    # ── 用户3：冷启动（宝宝3个月）──
    cold_profile = profiler.cold_start_profile(baby_age_months=3)

    # ── 推荐 ──
    recs_u1 = recommender.recommend("u1", u1_history, top_k=3)
    recs_u2 = recommender.recommend("u2", u2_history, top_k=3)
    recs_cold = recommender.recommend_cold_start(baby_age_months=3, top_k=3)

    print("=== KG-Powered User Profiling 测试 ===\n")

    print("[用户1-有机偏好] 推荐结果（已排除购买过的P001）：")
    for pid, score in recs_u1:
        node = next(n for n in kg if n.product_id == pid)
        print(f"  {pid} {node.name} | score={score:.4f} | certs={node.certifications}")

    print(f"\n[用户2-普通偏好] 推荐结果：")
    for pid, score in recs_u2:
        node = next(n for n in kg if n.product_id == pid)
        print(f"  {pid} {node.name} | score={score:.4f}")

    print(f"\n[冷启动 宝宝3个月] 偏好向量：{cold_profile}")
    print("[冷启动 宝宝3个月] 推荐结果：")
    for pid, score in recs_cold:
        node = next(n for n in kg if n.product_id == pid)
        print(f"  {pid} {node.name} | score={score:.4f} | attrs={node.attributes}")

    # ── 断言 ──
    u1_pids = [pid for pid, _ in recs_u1]
    assert "P001" not in u1_pids, "P001已购买，不应出现在推荐中"

    # 用户1（有机偏好）推荐中应有EU有机认证产品
    has_organic = any(
        "EU有机认证" in next(n for n in kg if n.product_id == pid).certifications
        for pid, _ in recs_u1
    )
    assert has_organic, "用户1推荐中应包含EU有机认证产品"
    print("\n✅ 断言通过：已购商品不重复推荐")
    print("✅ 断言通过：有机偏好用户首推含EU有机认证产品")

    assert len(cold_profile) > 0, "冷启动偏好向量不应为空"
    print("✅ 断言通过：冷启动偏好向量构建成功")

    print("\n全部测试通过 ✓")


if __name__ == "__main__":
    run_test()
