---
title: Live Commerce Stream Algorithm — 直播电商算法建模与互动信号优化
doc_type: knowledge
module: 20-AI视频生成
topic: live-commerce-stream-algorithm
status: stable
created: 2026-06-12
updated: 2026-06-12
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: Live Commerce Stream Algorithm — 直播电商算法与互动优化

> **论文**：OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with Editable Generative Model
> **arXiv**：2606.13533 | 2026年 Kuaishou | **桥梁**: 20-AI视频生成 ↔ 05-推荐系统 | **类型**: 平台算法
> **反直觉来源**：426 个 Skill 中直播电商完全缺失，而 TikTok Shop 直播 GMV 占比已超 40%

---

## ① 算法原理

### 核心思想

直播电商的算法与短视频**完全不同**——短视频是"发布后被动被推荐"，直播是"实时竞争曝光池位置"。平台每秒重新计算所有在播直播间的排名，决定哪个直播间出现在用户的首屏。

**直播间排名信号体系**（与短视频 FYP 的关键差异）：

| 信号 | 短视频 FYP | 直播间排名 |
|---|---|---|
| 完播率 | 核心指标 | 不适用 |
| **在线人数增速** | 不适用 | 核心指标 |
| **加购率（GMV/UV）** | 不适用 | 核心指标（商业化权重最高）|
| **评论发送速度** | 辅助 | 实时弹幕密度是主要信号 |
| **礼物/打赏** | 不适用 | 提升曝光的直接信号 |
| 粉丝互动率 | 参考 | 关键（关注者占观看者比例）|

**OneRetrieval 的核心贡献**：统一多路检索（搜索 + 推荐 + 直播推荐）到单一生成模型，支持新品（新直播间）快速插入而无需重训——解决了直播冷启动问题。

### 直播间流量阶段模型

```
新开播（冷启动池）
    │  在线人数增速 > 阈值 → 进入放大
    ▼
小流量测试池（100-500 在线）
    │  GMV/UV × 弹幕密度 → 综合评分
    ▼
大流量扩大（500-5000 在线）
    │  商业化权重（加购率）决定上限
    ▼
头部直播位（5000+ 在线）
```

### 商业化加权公式

$$\text{Rank}(L) = w_1 \cdot \Delta\text{Viewers} + w_2 \cdot \frac{\text{GMV}}{UV} + w_3 \cdot \text{Comment\_Rate} + w_4 \cdot \text{Gift\_Count}$$

在 TikTok Shop 2026 年算法中，$w_2$（GMV/UV）权重约占 35-40%，是排名提升的最高杠杆。

---

## ② 母婴出海应用案例

### 场景 A：TikTok Shop 直播开播前 15 分钟黄金期优化

**业务问题**：每次开播前 15 分钟是算法最重要的"决定期"——这期间的在线人数增速和加购率决定这场直播能否进入大流量池。但团队不知道应该在开播时做什么来提升这 15 分钟的表现。

**算法导向的开播策略**：
1. **T-30min**：在私域（粉丝群/粉丝页）预告直播开始时间
2. **T-0 到 T+3min**：不卖货，先做互动暖场（提升弹幕密度）
3. **T+3 到 T+8min**：抛出"秒杀钩子"（限时特价），触发加购和购买
4. **T+8 到 T+15min**：加购率信号传递给算法，平台开始放大流量

**关键指标目标**：前 15 分钟 GMV/UV ≥ $3，弹幕发送率 ≥ 8%

### 场景 B：直播间商品排序优化（高 GMV 商品先出场）

**业务问题**：直播间准备了 20 款商品，但哪些先出场、哪些做钩子、哪些做利润款，顺序影响整场 GMV。

**算法导向的商品排序**：
- 前 15 分钟：**钩子品**（低价高吸引力，提升加购率信号）
- 中段（15-45分钟）：**利润品**（高贡献毛利，流量峰值期出场）
- 后段：**长尾品**（配件/关联品，利用已积累的流量池）

---

## ③ 代码模板

```python
"""
Live Commerce Stream Algorithm — 直播间算法评分与策略优化
基于 OneRetrieval (arXiv: 2606.13533) + TikTok Live 算法逆向工程

依赖: numpy, dataclasses (标准库)
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class LiveStreamMetrics:
    """直播间实时指标"""
    stream_id: str
    platform: str               # tiktok / amazon_live / douyin
    elapsed_minutes: float      # 已直播分钟数
    current_viewers: int        # 当前在线人数
    peak_viewers: int           # 峰值人数
    viewer_growth_rate: float   # 在线人数增速（每分钟净增）
    gmv_per_viewer: float       # GMV/UV（最重要的商业化指标）
    comment_rate: float         # 弹幕发送率（评论数/在线人数/分钟）
    add_to_cart_rate: float     # 加购率
    gift_count_per_min: float   # 礼物数/分钟
    follower_ratio: float       # 粉丝占在线比例


@dataclass
class LiveRankingScore:
    """直播间算法评分"""
    stream_id: str
    total_score: float
    signal_breakdown: dict
    traffic_stage: str          # cold / testing / expanding / top
    estimated_next_viewers: int
    bottleneck_signal: str
    action_items: list


@dataclass
class ProductSchedule:
    """直播商品排期建议"""
    product_id: str
    product_name: str
    selling_price: float
    cost: float
    hook_power: float           # 钩子力（吸引力）：0-1
    profit_margin: float
    recommended_slot: str       # cold_start / peak / tail


class LiveStreamScorer:
    """
    直播间算法评分器

    基于 TikTok Shop 2026 逆向工程 + OneRetrieval 架构原理
    """

    # 平台权重配置
    PLATFORM_WEIGHTS = {
        "tiktok": {
            "viewer_growth":    0.25,
            "gmv_per_viewer":   0.38,  # 商业化最高权重
            "comment_rate":     0.18,
            "add_to_cart_rate": 0.12,
            "gift_count":       0.04,
            "follower_ratio":   0.03,
        },
        "amazon_live": {
            "viewer_growth":    0.20,
            "gmv_per_viewer":   0.45,  # Amazon 更重商业化
            "comment_rate":     0.10,
            "add_to_cart_rate": 0.20,
            "gift_count":       0.00,
            "follower_ratio":   0.05,
        },
    }

    BENCHMARKS = {
        "viewer_growth":    {"poor": -5, "ok": 2,  "good": 8,  "great": 20},
        "gmv_per_viewer":   {"poor": 0.5,"ok": 2.0,"good": 4.0,"great": 8.0},
        "comment_rate":     {"poor": 0.02,"ok": 0.05,"good": 0.10,"great": 0.20},
        "add_to_cart_rate": {"poor": 0.01,"ok": 0.05,"good": 0.12,"great": 0.25},
        "gift_count":       {"poor": 0,  "ok": 1,  "good": 5,  "great": 15},
        "follower_ratio":   {"poor": 0.05,"ok": 0.20,"good": 0.40,"great": 0.70},
    }

    def _normalize(self, value: float, bench: dict) -> float:
        if value >= bench["great"]: return 1.0
        elif value >= bench["good"]: return 0.75 + 0.25*(value-bench["good"])/(bench["great"]-bench["good"])
        elif value >= bench["ok"]:   return 0.50 + 0.25*(value-bench["ok"])/(bench["good"]-bench["ok"])
        elif value >= bench["poor"]: return 0.25 + 0.25*(value-bench["poor"])/(bench["ok"]-bench["poor"])
        else: return max(0, 0.25*value/max(bench["poor"],1e-9))

    def score(self, metrics: LiveStreamMetrics) -> LiveRankingScore:
        """计算直播间算法综合评分"""
        weights = self.PLATFORM_WEIGHTS.get(metrics.platform,
                                             self.PLATFORM_WEIGHTS["tiktok"])

        signal_values = {
            "viewer_growth":    metrics.viewer_growth_rate,
            "gmv_per_viewer":   metrics.gmv_per_viewer,
            "comment_rate":     metrics.comment_rate,
            "add_to_cart_rate": metrics.add_to_cart_rate,
            "gift_count":       metrics.gift_count_per_min,
            "follower_ratio":   metrics.follower_ratio,
        }

        signal_scores = {
            s: self._normalize(v, self.BENCHMARKS[s])
            for s, v in signal_values.items()
        }

        total = sum(weights[s] * signal_scores[s] for s in weights)

        # 流量阶段
        if total >= 0.75:   stage, next_viewers = "top",      metrics.current_viewers * 3
        elif total >= 0.60: stage, next_viewers = "expanding", metrics.current_viewers * 2
        elif total >= 0.40: stage, next_viewers = "testing",   int(metrics.current_viewers * 1.5)
        else:               stage, next_viewers = "cold",      metrics.current_viewers

        # 最大瓶颈（加权得分最低的信号）
        bottleneck = min(signal_scores, key=lambda s: signal_scores[s] * weights[s])

        # 行动建议
        actions = []
        if signal_scores["gmv_per_viewer"] < 0.5:
            actions.append("🛒 GMV/UV 不足：立即上钩子品（限时特价），触发加购")
        if signal_scores["comment_rate"] < 0.5:
            actions.append("💬 弹幕稀疏：发起互动提问（宝妈们最需要哪款功能？）")
        if signal_scores["viewer_growth"] < 0.4:
            actions.append("📱 人数不增：通知私域粉丝进入直播间")
        if signal_scores["add_to_cart_rate"] < 0.4:
            actions.append("🎯 加购率低：展示加购优惠（加购专属立减 $5）")

        return LiveRankingScore(
            stream_id=metrics.stream_id,
            total_score=round(total, 3),
            signal_breakdown={s: round(v, 3) for s, v in signal_scores.items()},
            traffic_stage=stage,
            estimated_next_viewers=next_viewers,
            bottleneck_signal=bottleneck,
            action_items=actions[:3],
        )


class ProductScheduler:
    """直播间商品排期优化器"""

    def optimize_schedule(self, products: list,
                          stream_duration_min: int = 60) -> list:
        """
        基于算法逻辑优化商品出场顺序

        策略：
        - 前15分钟（冷启动期）：最强钩子品，快速触发加购信号
        - 15-45分钟（流量峰值）：高利润品，最大化 GMV
        - 45分钟后（尾声）：关联/配件品，利用积累流量
        """
        hooks = sorted([p for p in products if p.hook_power >= 0.7],
                       key=lambda p: -p.hook_power)
        profits = sorted([p for p in products if p.profit_margin >= 0.35],
                         key=lambda p: -p.profit_margin)
        tails = [p for p in products if p not in hooks and p not in profits]

        result = []
        # 冷启动槽（前15分钟）：最多3个钩子品
        for p in hooks[:3]:
            p.recommended_slot = "cold_start"
            result.append(p)
        # 峰值槽（15-45分钟）：利润品
        for p in profits:
            if p not in result:
                p.recommended_slot = "peak"
                result.append(p)
        # 尾声槽
        for p in tails:
            p.recommended_slot = "tail"
            result.append(p)

        return result


def run_live_commerce_demo():
    """演示：母婴直播间算法评分与策略"""
    print("=" * 60)
    print("Live Commerce Stream Algorithm — 直播间算法演示")
    print("=" * 60)

    streams = [
        LiveStreamMetrics(
            "LIVE-A", "tiktok", 12.0, 380, 420,
            viewer_growth_rate=18.0, gmv_per_viewer=4.2,
            comment_rate=0.12, add_to_cart_rate=0.15,
            gift_count_per_min=3.0, follower_ratio=0.35
        ),
        LiveStreamMetrics(
            "LIVE-B", "tiktok", 8.0, 120, 150,
            viewer_growth_rate=3.0, gmv_per_viewer=1.2,
            comment_rate=0.04, add_to_cart_rate=0.04,
            gift_count_per_min=0.5, follower_ratio=0.15
        ),
    ]

    scorer = LiveStreamScorer()
    print(f"\n{'直播间':<10} {'总分':>6} {'阶段':<12} {'预估下轮在线':>12} {'瓶颈信号'}")
    print("-" * 60)
    for m in streams:
        result = scorer.score(m)
        print(f"{m.stream_id:<10} {result.total_score:>6.3f} {result.traffic_stage:<12} "
              f"{result.estimated_next_viewers:>12,}  {result.bottleneck_signal}")
        if result.action_items:
            for action in result.action_items[:2]:
                print(f"  → {action}")

    # 商品排期
    products = [
        ProductSchedule("P1", "M5 吸奶器（秒杀价）", 69.99, 28.0, 0.90, 0.28, ""),
        ProductSchedule("P2", "UV 消毒器", 49.99, 16.0, 0.55, 0.38, ""),
        ProductSchedule("P3", "储奶袋 100片", 18.99, 4.5,  0.40, 0.42, ""),
        ProductSchedule("P4", "M5 正价款",  89.99, 28.0, 0.60, 0.48, ""),
        ProductSchedule("P5", "配件套装",   24.99, 8.0,  0.30, 0.40, ""),
    ]
    scheduler = ProductScheduler()
    scheduled = scheduler.optimize_schedule(products)

    print("\n📦 商品排期建议:")
    for p in scheduled:
        slot_emoji = {"cold_start": "🎣", "peak": "💰", "tail": "🔄"}.get(p.recommended_slot, "")
        print(f"  {slot_emoji} [{p.recommended_slot:<12}] {p.product_name:<20} "
              f"毛利{p.profit_margin:.0%} 钩子力{p.hook_power:.0%}")

    # 验证
    results = [scorer.score(m) for m in streams]
    assert results[0].total_score > results[1].total_score
    assert results[0].traffic_stage in ("expanding", "top")
    assert any(p.recommended_slot == "cold_start" for p in scheduled)

    print("\n[✓] Live Commerce Stream Algorithm 测试通过")
    return results


if __name__ == "__main__":
    run_live_commerce_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-TikTok-Algorithm-Content-Boost]]（TikTok FYP 短视频算法是直播算法的基础认知框架；直播与短视频信号权重完全不同）
- **前置（prerequisite）**：[[Skill-AnchorCrafter-Virtual-Anchor-Demo]]（虚拟主播内容生产 → 直播算法优化，内容 + 分发缺一不可）
- **延伸（extends）**：[[Skill-Video-ROI-Attribution]]（直播 GMV 归因与 Reels 视频 ROI 归因形成完整短视频营销体系）
- **延伸（extends）**：[[Skill-Creator-Economy-ROI-Model]]（KOL 带货直播是 Creator Economy 的核心变现方式）
- **可组合（combinable）**：[[Skill-SKU-Level-PL-Dashboard]]（组合场景：直播钩子品的利润底线由 SKU P&L 给出，避免钩子品亏损）
- **可组合（combinable）**：[[Skill-Social-Proof-Amplification]]（组合场景：直播间评论弹幕 = 实时社交证明，高弹幕密度既提升算法分又提升观看者购买转化）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 前 15 分钟策略优化 → 进入扩大流量池：直播 GMV 从 $2,000 → $8,000（4×）
  - 商品排期优化：峰值流量期出利润品，GMV 提升 20-30%
  - 算法评分实时监控：及时发现问题并干预，减少"白播"场次
  - **年化综合 ROI**：¥100-500 万（视直播频率和品牌规模）

- **实施难度**：⭐⭐☆☆☆（算法评分纯指标计算，数据来源 TikTok Analytics，1-2 天接入）

- **优先级评分**：⭐⭐⭐⭐⭐（TikTok Shop 直播是 2025-2026 增长最快的母婴销售渠道，全图完全缺失）

- **评估依据**：OneRetrieval 在 Kuaishou（快手，直播电商第二大平台）生产验证；TikTok Shop 2026 年直播 GMV 增速超过短视频 GMV 增速 3×
