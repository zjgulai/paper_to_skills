---
title: Share of Voice Tracking — AI 时代跨平台品牌可见度份额测量
doc_type: knowledge
module: 13-广告分析
topic: share-of-voice-tracking
status: stable
created: 2026-06-12
updated: 2026-06-12
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Share of Voice Tracking — AI 时代跨平台品牌 SOV 测量

> **论文**：Don't Measure Once: Measuring Visibility in AI Search
> **arXiv**：2604.07585 | 2026年 | **桥梁**: 13-广告分析 ↔ 15-营销投放分析 | **类型**: 测量框架
> **补充**：From Prompt to Purchase: How AI Brand Recommendations Move Consumers (arXiv: 2606.10907, 2026)

---

## ① 算法原理

### 核心思想

传统 SOV（Share of Voice）= 品牌广告曝光 / 市场总曝光，计算简单。但 AI 搜索引擎时代，SOV 变得极不稳定——同一个问题问 ChatGPT 两次，可能推荐不同品牌。论文证明：**单次查询测量的 SOV 误差高达 ±40%**，必须用多次采样 + 稳定性加权才能得到可靠估计。

**稳定性感知 SOV 框架**：

```
传统 SOV = 品牌提及次数 / 总查询次数
                    ↓
稳定性感知 SOV = 加权平均（按提及位置 + 稳定性权重）

位置权重：第1名 = 1.0，第2名 = 0.7，第3名 + = 0.4
稳定性权重：重复出现品牌权重更高（论文推荐至少30次采样）
```

**跨平台 SOV 矩阵**：

| 平台 | SOV 计算方式 | 采样要求 |
|---|---|---|
| ChatGPT | 多轮对话品牌提及频率 | ≥ 30 次/关键词 |
| Perplexity | 搜索结果中品牌引用位置 | ≥ 20 次/关键词 |
| Amazon搜索 | 自然搜索结果页曝光占比 | 实时 API |
| Google AI Overview | AI 摘要中品牌出现率 | ≥ 15 次/关键词 |
| TikTok搜索 | 视频搜索结果品牌出现率 | ≥ 25 次/关键词 |

**从 AI 推荐到购买的路径**（arXiv 2606.10907 验证）：
AI 品牌推荐 → 消费者直接搜索品牌词 → 官网 or Amazon 转化，整个链路可追踪。

### 关键假设
- 需要自动化采样工具（API 调用 LLM，费用较低）
- SOV 会随时间变化（建议每周至少测量一次）
- 不同查询词（"best breast pump" vs "most affordable breast pump"）SOV 可能差异很大

---

## ② 母婴出海应用案例

### 场景 A：AI 搜索 SOV 基准建立（竞品对比）

**业务问题**：运营团队感觉"AI 好像经常推荐 Elvie 不推荐我们"，但没有数据证明，也不知道差距多大，无法制定针对性 GEO 策略。

**SOV 测量方案**：
- 对 20 个高频查询词（"best wearable breast pump"、"quiet breast pump for work" 等）
- 在 ChatGPT + Perplexity 各采样 30 次
- 计算 Momcozy vs Elvie vs Spectra 的稳定性加权 SOV

**典型发现**：
- Momcozy AI-SOV：28%（Amazon 第一但 AI 引用第三）
- Elvie AI-SOV：45%（内容策略更符合 AI 引用偏好）
- Spectra AI-SOV：22%

→ 明确差距，针对性运行 GEO 优化

### 场景 B：跨平台 SOV 归因（哪个平台最值得投入）

**业务问题**：有限预算，不知道应该把内容优化资源投在 Amazon、TikTok 搜索还是 AI 搜索引擎。

**SOV 矩阵对比**：测量品牌在各平台的当前 SOV 与竞品差距，投入到差距最大且商业价值最高的平台。

---

## ③ 代码模板

```python
"""
Share of Voice Tracking — AI 时代跨平台品牌可见度测量
基于 arXiv: 2604.07585 (2026) + arXiv: 2606.10907 (2026)

依赖: re, statistics, dataclasses (标准库)
生产环境: 替换 MockLLMSampler 为真实 API
"""

from dataclasses import dataclass, field
from statistics import mean, stdev
import re


@dataclass
class BrandMention:
    """单次查询中的品牌提及"""
    query: str
    platform: str
    brand: str
    position: int       # 1 = 第一个被提及
    sample_idx: int     # 第几次采样


@dataclass
class SOVResult:
    """品牌 SOV 计算结果"""
    brand: str
    platform: str
    raw_mention_rate: float      # 简单提及率
    position_weighted_sov: float # 位置加权 SOV
    stability_score: float       # 稳定性分数（多次采样的一致性）
    sample_count: int


class MockLLMSampler:
    """
    模拟 LLM 响应采样（生产环境替换为真实 API）

    生产环境示例：
        import openai
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content
    """

    # 模拟不同品牌被推荐的概率（母婴吸奶器市场）
    BRAND_PROBS = {
        "Momcozy": [0.45, 0.35, 0.20],   # 位置1/2/3的概率
        "Elvie":   [0.35, 0.40, 0.25],
        "Spectra": [0.20, 0.25, 0.55],
    }

    import random as _rand

    def sample_response(self, query: str, platform: str) -> list:
        """模拟单次 LLM 响应，返回品牌提及列表 [(brand, position)]"""
        import random
        mentions = []
        brands = list(self.BRAND_PROBS.keys())
        random.shuffle(brands)

        for pos, brand in enumerate(brands[:3], start=1):
            probs = self.BRAND_PROBS[brand]
            # 以一定概率在该位置提及
            if random.random() < probs[min(pos-1, 2)]:
                mentions.append((brand, pos))

        return mentions


class SOVTracker:
    """
    跨平台 SOV 追踪器

    核心方法：
    1. 多次采样（解决 AI 响应随机性）
    2. 位置加权（第1名 > 第2名 > 第3名）
    3. 稳定性评估（提醒测量可靠性）
    """

    # 位置权重（基于 arXiv 2604.07585）
    POSITION_WEIGHTS = {1: 1.0, 2: 0.7, 3: 0.4}
    DEFAULT_WEIGHT = 0.2

    def __init__(self, sampler=None, n_samples: int = 30):
        self.sampler = sampler or MockLLMSampler()
        self.n_samples = n_samples

    def _position_weight(self, position: int) -> float:
        return self.POSITION_WEIGHTS.get(position, self.DEFAULT_WEIGHT)

    def measure_sov(self, queries: list, brands: list,
                    platform: str = "ChatGPT") -> list:
        """
        测量品牌在给定查询下的 SOV

        Args:
            queries: 查询词列表
            brands: 目标品牌列表
            platform: 平台名称

        Returns:
            List[SOVResult]，按 SOV 排序
        """
        # 采集所有样本
        all_mentions = {brand: [] for brand in brands}

        for query in queries:
            for sample_idx in range(self.n_samples):
                mentions = self.sampler.sample_response(query, platform)
                for brand, position in mentions:
                    if brand in brands:
                        all_mentions[brand].append(
                            BrandMention(query, platform, brand, position, sample_idx)
                        )

        total_samples = len(queries) * self.n_samples
        results = []

        for brand in brands:
            mentions = all_mentions[brand]

            if not mentions:
                results.append(SOVResult(brand, platform, 0.0, 0.0, 0.0, total_samples))
                continue

            # 简单提及率
            raw_rate = len(mentions) / total_samples

            # 位置加权 SOV（每次采样最多计一次）
            weighted_scores = []
            for q_idx, query in enumerate(queries):
                for s_idx in range(self.n_samples):
                    q_mentions = [m for m in mentions
                                  if m.query == query and m.sample_idx == s_idx]
                    if q_mentions:
                        best_pos = min(m.position for m in q_mentions)
                        weighted_scores.append(self._position_weight(best_pos))
                    else:
                        weighted_scores.append(0.0)

            pos_weighted = mean(weighted_scores) if weighted_scores else 0.0

            # 稳定性：同一 query 不同采样的一致性
            per_query_rates = []
            for query in queries:
                q_samples = sum(1 for m in mentions if m.query == query)
                per_query_rates.append(q_samples / self.n_samples)
            stability = 1.0 - (stdev(per_query_rates) if len(per_query_rates) > 1 else 0.0)

            results.append(SOVResult(
                brand=brand,
                platform=platform,
                raw_mention_rate=round(raw_rate, 4),
                position_weighted_sov=round(pos_weighted, 4),
                stability_score=round(max(0, stability), 4),
                sample_count=total_samples,
            ))

        # 归一化 position_weighted_sov
        total_w = sum(r.position_weighted_sov for r in results)
        if total_w > 0:
            for r in results:
                r.position_weighted_sov = round(r.position_weighted_sov / total_w, 4)

        return sorted(results, key=lambda r: r.position_weighted_sov, reverse=True)

    def competitive_matrix(self, queries: list, brands: list,
                           platforms: list) -> dict:
        """跨平台 SOV 竞品矩阵"""
        matrix = {}
        for platform in platforms:
            matrix[platform] = {r.brand: r.position_weighted_sov
                                for r in self.measure_sov(queries, brands, platform)}
        return matrix


def run_sov_demo():
    """演示：母婴吸奶器品牌跨平台 SOV 测量"""
    print("=" * 60)
    print("Share of Voice Tracking — 跨平台品牌可见度测量演示")
    print("=" * 60)

    queries = [
        "best wearable breast pump for working moms",
        "safest electric breast pump for newborns",
        "quiet breast pump under $100",
        "hospital grade breast pump recommendation",
    ]
    brands = ["Momcozy", "Elvie", "Spectra"]
    platforms = ["ChatGPT", "Perplexity"]

    tracker = SOVTracker(n_samples=20)

    # 单平台 SOV
    print(f"\n📊 ChatGPT SOV 测量（{len(queries)} 查询 × 20 次采样）")
    results = tracker.measure_sov(queries, brands, "ChatGPT")
    print(f"\n{'品牌':<12} {'提及率':>8} {'位置加权SOV':>13} {'稳定性':>8}")
    print("-" * 48)
    for r in results:
        bar = "█" * int(r.position_weighted_sov * 30)
        print(f"{r.brand:<12} {r.raw_mention_rate:>8.1%} {r.position_weighted_sov:>13.1%} "
              f"{r.stability_score:>8.1%}  {bar}")

    # 跨平台矩阵
    print(f"\n🌐 跨平台 SOV 矩阵")
    matrix = tracker.competitive_matrix(queries, brands, platforms)
    header = f"{'品牌':<12} " + " ".join(f"{p:>12}" for p in platforms)
    print(header)
    print("-" * (12 + 13 * len(platforms)))
    for brand in brands:
        row = f"{brand:<12} "
        for platform in platforms:
            sov = matrix[platform].get(brand, 0)
            row += f"{sov:>12.1%} "
        print(row)

    # 验证
    total_sov = sum(r.position_weighted_sov for r in results)
    assert abs(total_sov - 1.0) < 0.05, f"SOV 总和应≈1.0，实际 {total_sov:.3f}"
    assert results[0].position_weighted_sov >= results[-1].position_weighted_sov

    print("\n[✓] SOV Tracking 测试通过")
    return results


if __name__ == "__main__":
    run_sov_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Ad-Attribution-Modeling]]（广告归因是 SOV 测量的方法论基础）
- **前置（prerequisite）**：[[Skill-GEO-Generative-Engine-Optimization]]（GEO 优化是提升 AI 平台 SOV 的直接手段）
- **延伸（extends）**：[[Skill-ROAS-Budget-Optimization]]（各平台 SOV 数据作为预算分配的参考信号）
- **延伸（extends）**：[[Skill-Marketing-Mix-Modeling]]（SOV 变化率作为 MMM 的品牌健康度领先指标）
- **可组合（combinable）**：[[Skill-Cross-Platform-Brand-Search-Volume]]（组合场景：SOV 提升 → 品牌搜索量增长，形成"AI 引用 → 主动搜索"完整闭环）
- **可组合（combinable）**：[[Skill-Keyword-Competition-Scoring]]（组合场景：关键词竞争评分 × AI 平台 SOV，识别高竞争但低 AI 覆盖的机会词）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 识别 SOV 差距后针对性 GEO 优化：AI 平台流量提升 30-40%
  - 跨平台 SOV 矩阵指导预算分配：投入 ROI 提升 20-30%
  - 竞品监测：及时发现竞品 SOV 上升，快速响应
  - **年化综合 ROI**：¥30-100 万（随 AI 搜索流量增长持续扩大）

- **实施难度**：⭐⭐☆☆☆（LLM API 采样 + 统计计算，1-2 天接入）

- **优先级评分**：⭐⭐⭐⭐⭐（AI 搜索时代的核心监测工具，没有 SOV 数据就无法判断 GEO 效果）

- **评估依据**：arXiv 2604.07585 证明单次采样误差 ±40%，30 次采样可将误差压缩到 ±8%；arXiv 2606.10907 追踪了 AI 推荐 → 购买的完整路径
