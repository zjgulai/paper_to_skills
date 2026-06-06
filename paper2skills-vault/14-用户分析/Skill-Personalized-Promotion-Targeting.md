---
title: Personalized Promotion Targeting — 个性化促销定向：用户响应异质性建模
doc_type: knowledge
module: 14-用户分析
topic: personalized-promotion-targeting-heterogeneous
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: Personalized Promotion Targeting — 个性化促销定向

---

## ① 算法原理

### 核心问题：促销"一刀切"的浪费

传统促销策略对所有用户发放相同折扣，造成两类浪费：
1. **Cannibalization（自相残杀）**：把折扣发给"本来就会买"的高意愿用户，白白损失利润
2. **无效投放**：对低响应用户发促销，浪费预算但无法提升转化

### 异质性处理效应（HTE）的业务含义

每个用户对促销的响应是不同的。HTE 将用户分为4类（Uplift 四象限）：
- **Persuadables（真目标）**：有促销才买，无促销不买 → 精准投放
- **Sure Things（本来就买）**：有无促销都会买 → 停止发券，节省成本
- **Lost Causes（无效投放）**：有无促销都不买 → 跳过
- **Sleeping Dogs（反效果）**：发促销反而降低购买意愿 → 绝对不投

### 基于用户分群的差异化促销策略

通过 RFM、月龄、行为特征将用户分群，对每个分群估计响应概率 $p(buy | promotion, segment)$ 和期望增量价值 $\Delta V(segment)$。

**最优分配目标**：在预算约束下最大化总增量价值：

$$\max \sum_{i \in U} x_i \cdot \Delta V_i \cdot p_i$$
$$\text{s.t.} \sum_{i \in U} x_i \cdot c_i \leq B, \quad x_i \in \{0, 1\}$$

其中 $x_i$ 为是否发放促销，$c_i$ 为促销成本，$B$ 为预算上限。

### 成本约束下的 Knapsack 最优分配

上述问题是标准的 0-1 背包问题。对连续松弛版本（Fractional Knapsack），按 **ROI 比率**（$\Delta V_i \cdot p_i / c_i$）降序贪心排列，在预算耗尽前优先分配给 ROI 最高的分群。0-1 整数版本用动态规划精确求解。

### 避免 Cannibalization

关键：需要**增量响应概率** $\Delta p_i = p(buy | promo, i) - p(buy | no\_promo, i)$，而非绝对响应概率。分群响应率高 ≠ 增量高，"Sure Things"的绝对响应率很高但增量接近 0。

---

## ② 母婴出海应用案例

### 场景1：母婴奶粉换购活动定向

**业务问题**：品牌推出"Stage1→Stage2奶粉换购"优惠，Stage2 售价更高，换购成功即提升 LTV。但如果发给所有 Stage1 用户，大量原本会自然升阶的用户白白享受了折扣。

**应用流程**：
1. **用户分群**：基于宝宝月龄（4月龄以下/4-5月龄/6月龄以上）+ RFM 分群
2. **关键洞察**：6月龄以上用户已自然升阶（Sure Things），4-5月龄用户是最佳目标（Persuadables）
3. **响应模型**：历史数据估计每群的增量响应概率 $\Delta p_i$
4. **Knapsack 分配**：在促销预算下，优先分配给 ROI = $\Delta p_i \times \Delta LTV_i / cost_i$ 最高的分群

**预期产出**：
- 仅投放 4-5 月龄用户（30% 的活跃 Stage1 用户），节省 70% 促销成本
- 换购率：从全体投放的 12% 提升至精准投放的 28%
- 人均促销成本降低 40%，整体 ROI 提升 2.3x

**关键规则**：宝宝月龄通过注册时填写的预产期/出生日期推断，每月自动更新分群。

### 场景2：WF-C 挽留促销精准投放

**业务问题**：WF-C 工作流针对即将流失用户发挽留促销。但"即将流失"≠"一定会流失"，且不同价值的用户需要不同的挽留策略，否则高成本挽留手段（专属客服）被低价值用户消耗。

**分群策略矩阵**：

| 用户价值 | 流失概率 | 响应概率 | 策略 | 预算 |
|---|---|---|---|---|
| 高价值（LTV>$500） | 高 | 高 | 专属客服一对一 | $50/人 |
| 中价值（LTV $100-500） | 高 | 中 | 专属优惠券 10% off | $15/人 |
| 低价值（LTV<$100） | 高 | 低 | 自助式邮件优惠 | $2/人 |
| Sleeping Dogs | 任意 | 负增量 | 不干预 | $0 |

**应用流程**：
1. **流失意图检测**：基于登录频率、浏览时长、购物车放弃率
2. **价值分层**：12月 LTV 预测分档
3. **Knapsack 优化**：月度挽留预算 $50,000，最大化 Retained LTV

**预期产出**：挽留 ROI 从 1.8x 提升至 3.2x；专属客服资源集中在真正高价值用户（节省 60% 客服时间）

---

## ③ 代码模板

```python
"""
Personalized Promotion Targeting — 个性化促销定向
异质性响应建模 + Knapsack 预算优化

纯 Python 标准库，无 sklearn/pandas 依赖
Python 3.14 兼容
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


# ─── 数据结构 ────────────────────────────────────────────────────────────────

@dataclass
class UserSegment:
    """用户分群的促销响应特征"""
    segment_id: str
    segment_name: str
    n_users: int                    # 该分群用户数
    propensity_to_respond: float    # 绝对响应概率 P(buy | promo)
    baseline_response: float        # 无促销响应概率 P(buy | no promo)
    expected_value: float           # 响应后期望 LTV 增量（$）
    cost: float                     # 人均促销成本（$）

    @property
    def incremental_response(self) -> float:
        """增量响应概率 = 有促销 - 无促销（排除 Sure Things）"""
        return max(0.0, self.propensity_to_respond - self.baseline_response)

    @property
    def roi_per_dollar(self) -> float:
        """每美元促销成本的期望增量价值"""
        if self.cost <= 0:
            return 0.0
        return (self.incremental_response * self.expected_value) / self.cost

    @property
    def segment_type(self) -> str:
        """自动识别象限类型"""
        if self.incremental_response < 0.02:
            if self.baseline_response > 0.5:
                return "Sure Things"
            return "Lost Causes / Sleeping Dogs"
        if self.roi_per_dollar < 0.5:
            return "Low ROI Persuadables"
        return "Persuadables"


@dataclass
class AllocationResult:
    """分配结果"""
    segment: UserSegment
    allocated_users: int
    allocation_fraction: float      # 该分群中被分配的比例
    expected_incremental_ltv: float
    total_cost: float


# ─── 异质性响应建模 ──────────────────────────────────────────────────────────

class HeterogeneousResponseModeler:
    """
    用户分群异质性响应建模器
    基于历史实验数据或 Uplift 模型输出，估计每个分群的响应特征
    """

    def __init__(self) -> None:
        self._segments: list[UserSegment] = []

    def fit(self, segments: list[UserSegment]) -> "HeterogeneousResponseModeler":
        """加载分群响应数据"""
        self._segments = sorted(segments, key=lambda s: s.roi_per_dollar, reverse=True)
        return self

    def predict_response_probability(self, segment_id: str) -> dict[str, float]:
        """预测指定分群的响应概率分布"""
        for seg in self._segments:
            if seg.segment_id == segment_id:
                return {
                    "propensity_with_promo": seg.propensity_to_respond,
                    "baseline_no_promo": seg.baseline_response,
                    "incremental_lift": seg.incremental_response,
                    "roi_per_dollar": seg.roi_per_dollar,
                    "segment_type": seg.segment_type,
                }
        raise ValueError(f"分群 {segment_id!r} 不存在")

    def print_segment_report(self) -> None:
        """打印分群响应报告（按 ROI 降序）"""
        print(f"\n{'分群':>15} {'类型':>20} {'增量响应':>8} {'ROI/$':>8} {'人均成本':>8}")
        print("-" * 65)
        for seg in self._segments:
            print(
                f"{seg.segment_name:>15} {seg.segment_type:>20} "
                f"{seg.incremental_response:>8.3f} {seg.roi_per_dollar:>8.2f} "
                f"${seg.cost:>6.1f}"
            )

    @property
    def segments(self) -> list[UserSegment]:
        return self._segments


# ─── Knapsack 预算优化分配 ───────────────────────────────────────────────────

class PromotionAllocator:
    """
    在预算约束下最大化总增量 LTV 的促销分配器

    策略：
    1. 按 ROI/$ 降序排列分群（贪心连续 Knapsack）
    2. 过滤掉 Sure Things / Sleeping Dogs（增量 < 阈值）
    3. 在预算内逐分群分配
    """

    def __init__(
        self,
        budget: float,
        min_incremental_response: float = 0.02,
    ) -> None:
        self.budget = budget
        self.min_incremental_response = min_incremental_response

    def allocate(self, segments: list[UserSegment]) -> list[AllocationResult]:
        """
        贪心 Fractional Knapsack 分配
        对每个分群，可选择分配 0~100% 的用户
        """
        # 按 ROI 降序，过滤无效分群
        eligible = [
            s for s in segments
            if s.incremental_response >= self.min_incremental_response and s.cost > 0
        ]
        eligible.sort(key=lambda s: s.roi_per_dollar, reverse=True)

        results: list[AllocationResult] = []
        remaining_budget = self.budget

        for seg in eligible:
            if remaining_budget <= 0:
                break

            max_cost_for_segment = seg.n_users * seg.cost
            allocatable_cost = min(max_cost_for_segment, remaining_budget)
            fraction = allocatable_cost / max_cost_for_segment
            allocated_users = int(seg.n_users * fraction)

            if allocated_users == 0:
                continue

            actual_cost = allocated_users * seg.cost
            incremental_ltv = allocated_users * seg.incremental_response * seg.expected_value

            results.append(AllocationResult(
                segment=seg,
                allocated_users=allocated_users,
                allocation_fraction=round(fraction, 4),
                expected_incremental_ltv=round(incremental_ltv, 2),
                total_cost=round(actual_cost, 2),
            ))
            remaining_budget -= actual_cost

        return results

    def max_roi_allocate(self, segments: list[UserSegment]) -> list[AllocationResult]:
        """同 allocate，别名方便调用"""
        return self.allocate(segments)

    def print_allocation_report(self, results: list[AllocationResult]) -> None:
        """打印分配结果"""
        total_ltv = sum(r.expected_incremental_ltv for r in results)
        total_cost = sum(r.total_cost for r in results)
        total_users = sum(r.allocated_users for r in results)

        print(f"\n{'分群':>15} {'分配用户':>8} {'分配比例':>8} {'成本':>10} {'增量LTV':>12}")
        print("-" * 60)
        for r in results:
            print(
                f"{r.segment.segment_name:>15} {r.allocated_users:>8,d} "
                f"{r.allocation_fraction:>8.1%} ${r.total_cost:>9,.0f} "
                f"${r.expected_incremental_ltv:>11,.0f}"
            )
        print("-" * 60)
        print(f"{'合计':>15} {total_users:>8,d} {'':>8} ${total_cost:>9,.0f} ${total_ltv:>11,.0f}")
        overall_roi = total_ltv / total_cost if total_cost > 0 else 0
        print(f"\n整体 ROI: {overall_roi:.2f}x （预算: ${self.budget:,.0f}，已用: ${total_cost:,.0f}）")


# ─── 测试 ────────────────────────────────────────────────────────────────────

def _build_test_segments() -> list[UserSegment]:
    """5 个用户群：覆盖全部4种象限类型"""
    return [
        UserSegment(
            segment_id="S1",
            segment_name="4-5月龄换购",
            n_users=5000,
            propensity_to_respond=0.45,
            baseline_response=0.08,  # 增量 0.37 — Persuadables
            expected_value=120.0,
            cost=15.0,
        ),
        UserSegment(
            segment_id="S2",
            segment_name="6月+自然升阶",
            n_users=8000,
            propensity_to_respond=0.80,
            baseline_response=0.78,  # 增量 0.02 — Sure Things
            expected_value=80.0,
            cost=15.0,
        ),
        UserSegment(
            segment_id="S3",
            segment_name="高价值挽留",
            n_users=1000,
            propensity_to_respond=0.60,
            baseline_response=0.10,  # 增量 0.50 — 高 ROI Persuadables
            expected_value=500.0,
            cost=50.0,
        ),
        UserSegment(
            segment_id="S4",
            segment_name="中价值挽留",
            n_users=3000,
            propensity_to_respond=0.35,
            baseline_response=0.05,  # 增量 0.30 — 中 ROI Persuadables
            expected_value=150.0,
            cost=15.0,
        ),
        UserSegment(
            segment_id="S5",
            segment_name="低价值沉默",
            n_users=10000,
            propensity_to_respond=0.05,
            baseline_response=0.04,  # 增量 0.01 — Lost Causes
            expected_value=30.0,
            cost=2.0,
        ),
    ]


def main() -> None:
    print("=" * 60)
    print("Loop 51-B: Personalized Promotion Targeting — 验证")
    print("=" * 60)

    segments = _build_test_segments()

    # ─── 响应建模 ───
    modeler = HeterogeneousResponseModeler()
    modeler.fit(segments)
    modeler.print_segment_report()

    # 验证 Sure Things 识别
    s2_info = modeler.predict_response_probability("S2")
    assert s2_info["segment_type"] == "Sure Things", f"S2 应为 Sure Things: {s2_info['segment_type']}"
    print("\n✅ Sure Things 正确识别（6月+自然升阶用户）")

    # 验证 Persuadables ROI 最高
    s3_roi = modeler.predict_response_probability("S3")["roi_per_dollar"]
    s5_roi = modeler.predict_response_probability("S5")["roi_per_dollar"]
    assert s3_roi > s5_roi, f"高价值挽留 ROI 应高于低价值: {s3_roi:.3f} vs {s5_roi:.3f}"
    print(f"✅ ROI 排序正确: 高价值挽留 {s3_roi:.2f}x > 低价值沉默 {s5_roi:.2f}x")

    # ─── Knapsack 分配 ───
    print("\n" + "=" * 60)
    print("预算约束下最优分配（预算: $80,000）")
    print("=" * 60)

    allocator = PromotionAllocator(budget=80_000.0, min_incremental_response=0.02)
    results = allocator.max_roi_allocate(segments)
    allocator.print_allocation_report(results)

    # 验证：Sure Things 不应被分配
    allocated_ids = {r.segment.segment_id for r in results}
    assert "S2" not in allocated_ids, "Sure Things（S2）不应被分配促销"
    print("\n✅ Sure Things（6月+用户）已被过滤，不发促销")

    # 验证：总成本 ≤ 预算
    total_cost = sum(r.total_cost for r in results)
    assert total_cost <= 80_000.0 + 1.0, f"超预算: ${total_cost:,.0f}"
    print(f"✅ 预算约束满足: 已用 ${total_cost:,.0f} ≤ $80,000")

    # 验证：ROI > 1.0
    total_ltv = sum(r.expected_incremental_ltv for r in results)
    roi = total_ltv / total_cost if total_cost > 0 else 0
    assert roi > 1.0, f"整体 ROI 应 > 1.0: {roi:.2f}"
    print(f"✅ 整体 ROI 为正: {roi:.2f}x")
    print("\n✅ 所有验证通过 — Loop 51-B Personalized Promotion Targeting")


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

### 前置技能
- [[Skill-RFM-Customer-Segmentation]] — RFM 分群是本技能用户分群的主要输入
- [[Skill-User-Lifecycle-STAN]] — 用户生命周期阶段影响响应概率估计
- [[Skill-Guardrailed-Uplift-Targeting]] — 提供增量响应概率的底层估计方法

### 延伸技能
- [[Skill-Causal-Cohort-Analysis]] — 因果队列分析验证分配策略的长期 LTV 效果
- [[Skill-Shopping-Companion-Agent]] — Agent 实时触发个性化促销投放

### 可组合
- [[Skill-AIGP-LLM-Dynamic-Pricing]] — 动态定价 + 个性化促销形成完整价格策略
- [[Skill-AgeMem-Unified-Agent-Memory]] — Agent 记忆层支持月龄感知的精准触达时机

---

## ⑤ 商业价值评估

### ROI 预估

**场景1（母婴换购精准定向）**：促销成本降低 40%，换购率从 12% 提升至 28%；年化节省促销预算 300-600 万元；整体 ROI 提升 2.3x。

**场景2（挽留促销优化）**：挽留 ROI 从 1.8x 提升至 3.2x；客服资源聚焦高价值用户，人效提升 60%；年化挽留收益增加 150-300 万元。

### 实施难度：⭐⭐⭐☆☆ (3/5)

- 易处：Knapsack 贪心算法简单高效；分群可复用现有 RFM 逻辑
- 难处：增量响应概率需要历史 A/B 数据或 Uplift 模型支撑；分群定义依赖业务知识
- 前提：需要至少1次历史A/B实验或准实验数据来估计基线响应率

### 优先级评分：⭐⭐⭐⭐⭐ (5/5)

**评估依据**：
1. **直接解决促销 Cannibalization 痛点**，适用于所有促销场景
2. **组合价值高**：与 RFM、Uplift、因果队列形成完整促销决策闭环
3. **实施门槛低**：不需要复杂 ML 模型，基于规则 + Knapsack 即可部署
4. **业务影响即时**：首次上线可在2-4周内看到明显 ROI 提升
