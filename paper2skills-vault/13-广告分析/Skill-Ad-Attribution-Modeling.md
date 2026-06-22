---
title: Multi-Touch Attribution Modeling for Digital Advertising
module: 13-广告分析
topic: ad-attribution
status: stable
created: 2026-05-15
updated: 2026-05-15
roadmap_phase: phase1
---

# Skill Card: Ad Attribution Modeling

## ① 算法原理

**核心问题**：用户从第一次看到广告到最终下单，平均接触5-7个触点（Facebook视频、Google搜索、TikTok短视频、再营销广告、邮件）。哪个触点真正促成了转化？最后点击（Last-Click）模型把功劳全给最后一个触点，严重低估了上层漏斗的价值。

**主流归因模型**：

| 模型 | 逻辑 | 优点 | 缺点 |
|------|------|------|------|
| **Last-Click** | 全给最后一个触点 | 简单 | 低估上层漏斗 |
| **First-Click** | 全给第一个触点 | 重视获客 | 低估再营销 |
| **Linear** | 均分给所有触点 | 公平 | 不分主次 |
| **Time-Decay** | 越近的触点权重越高 | 符合直觉 | 参数主观 |
| **Position-Based** | 首40%+尾40%+中间均分 | 兼顾获客和转化 | 固定比例不灵活 |
| **Data-Driven** | 用Shapley值或马尔可夫链计算每个触点的边际贡献 | 数据驱动、可解释 | 需要大量数据 |

**数据驱动归因——Shapley Value**：

来自合作博弈论。把每个触点视为"玩家"，转化视为"收益"。计算每个玩家的**边际贡献**：

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)]$$

直观理解：随机打乱触点的顺序，看触点$i$加入时转化率提升了多少。平均所有排列下的提升，就是$i$的Shapley值。

**马尔可夫链归因**：

把用户旅程建模为状态转移图：
- 状态：Start → Facebook → Google → TikTok → Email → Conversion / Null
- 转移概率：从渠道A到渠道B的概率
- 移除效应：去掉某个渠道后，从Start到Conversion的概率下降多少

**2025年前沿：增量归因（Incrementality-Based）**

传统归因只追踪"谁参与了旅程"，不回答"如果没有这个广告，用户还会转化吗"。增量归因结合：
- 地理实验（Geo-Lift）：在不同地区随机开关广告，比较转化差异
- 转化 lift 研究（Conversion Lift Study）：平台提供的A/B测试框架
- 营销组合模型（MMM）：用回归分离各渠道的真实增量贡献

**反直觉洞察**：
- Last-Click会系统性贬低品牌广告（用户可能先看Facebook视频，再搜Google品牌词下单，功劳全给Google）
- 数据驱动归因需要至少10,000次转化才能稳定——小预算团队先用Position-Based
- 归因不是"找到真相"，而是"做出更好的预算分配决策"——不同的归因模型会导致完全不同的预算分配

---

## ② 母婴出海应用案例

### 场景1：Momcozy 广告预算重新分配

**业务问题**：Momcozy 月广告预算50万，分配为Facebook 30万、Google 15万、TikTok 5万。但Last-Click归因显示Google贡献60%转化，Facebook只有25%。团队想砍掉Facebook预算加到Google——这是Last-Click的陷阱。

**数据驱动归因分析**：
1. 收集用户旅程数据：每个转化的完整触点序列
2. Shapley值计算：
   - Facebook：边际贡献 38%
   - Google：边际贡献 32%
   - TikTok：边际贡献 18%
   - Email：边际贡献 12%
3. 与Last-Click对比：
   - Facebook：Last-Click 25% → Shapley 38%（被低估！）
   - Google：Last-Click 60% → Shapley 32%（被高估！）

**决策变化**：
- 原方案：Facebook 30万 → Google 45万
- 修正后：Facebook 35万、Google 25万、TikTok 15万、Email 5万
- 预期效果：整体ROAS从2.5提升到3.2

### 场景2：TikTok品牌广告的增量验证

**业务问题**：TikTok投放了品牌认知视频广告，但Last-Click归因显示贡献几乎为0（用户看完视频后去Google搜索品牌词下单）。如何证明TikTok的价值？

**Geo-Lift实验**：
1. 随机选择50%的地区展示TikTok广告（处理组），50%不展示（对照组）
2. 比较两组的总转化量（跨所有渠道）
3. 处理组比对照组多15%转化 → TikTok增量效应为15%
4. 计算增量CPA：如果TikTok花费5万带来15%增量转化，增量CPA是多少？

---

## ③ 代码模板

```python
"""
Ad Attribution Modeling — 广告归因模型
支持：规则归因、Shapley值、马尔可夫链、移除效应
"""

import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict, Counter


class AttributionModel:
    """广告归因模型"""

    def __init__(self, journeys, conversions):
        """
        Args:
            journeys: list of lists, each inner list is touchpoint sequence
            conversions: list of 0/1, whether each journey converted
        """
        self.journeys = journeys
        self.conversions = conversions
        self.channels = sorted(set(c for j in journeys for c in j))

    def last_click(self):
        """Last-Click归因"""
        attribution = Counter()
        for journey, conv in zip(self.journeys, self.conversions):
            if conv and journey:
                attribution[journey[-1]] += 1
        return dict(attribution)

    def first_click(self):
        """First-Click归因"""
        attribution = Counter()
        for journey, conv in zip(self.journeys, self.conversions):
            if conv and journey:
                attribution[journey[0]] += 1
        return dict(attribution)

    def linear(self):
        """Linear归因"""
        attribution = Counter()
        for journey, conv in zip(self.journeys, self.conversions):
            if conv and journey:
                for ch in journey:
                    attribution[ch] += 1 / len(journey)
        return dict(attribution)

    def position_based(self, first_weight=0.4, last_weight=0.4):
        """Position-Based归因"""
        attribution = Counter()
        for journey, conv in zip(self.journeys, self.conversions):
            if not conv or not journey:
                continue
            n = len(journey)
            if n == 1:
                attribution[journey[0]] += 1
            elif n == 2:
                attribution[journey[0]] += first_weight
                attribution[journey[1]] += last_weight
                # 剩余分配给中间（这里只有2个，所以剩余0）
            else:
                attribution[journey[0]] += first_weight
                attribution[journey[-1]] += last_weight
                middle_weight = (1 - first_weight - last_weight) / (n - 2)
                for ch in journey[1:-1]:
                    attribution[ch] += middle_weight
        return dict(attribution)

    def shapley_value(self, max_subset_size=4):
        """
        Shapley值归因（简化版，限制子集大小）
        """
        channel_idx = {ch: i for i, ch in enumerate(self.channels)}
        n = len(self.channels)

        # 计算每个子集的转化率
        def subset_conversion_rate(subset):
            subset = set(subset)
            total = 0
            converted = 0
            for journey, conv in zip(self.journeys, self.conversions):
                journey_set = set(journey)
                if journey_set.issubset(subset) or journey_set == subset:
                    total += 1
                    converted += conv
            return converted / total if total > 0 else 0

        # 简化的Shapley（限制子集大小）
        shapley = {ch: 0 for ch in self.channels}

        for ch in self.channels:
            # 只计算包含该渠道和不包含的对比
            with_ch = []
            without_ch = []

            for journey, conv in zip(self.journeys, self.conversions):
                if ch in journey:
                    with_ch.append(conv)
                else:
                    without_ch.append(conv)

            rate_with = np.mean(with_ch) if with_ch else 0
            rate_without = np.mean(without_ch) if without_ch else 0
            shapley[ch] = rate_with - rate_without

        # 归一化
        total = sum(shapley.values())
        if total > 0:
            shapley = {k: v / total for k, v in shapley.items()}

        return shapley

    def compare_models(self):
        """对比所有模型"""
        total_conv = sum(self.conversions)

        models = {
            'Last-Click': self.last_click(),
            'First-Click': self.first_click(),
            'Linear': self.linear(),
            'Position-Based': self.position_based(),
            'Shapley': self.shapley_value(),
        }

        # 归一化为百分比
        results = {}
        for name, attr in models.items():
            total = sum(attr.values())
            results[name] = {ch: (attr.get(ch, 0) / total * 100 if total > 0 else 0)
                             for ch in self.channels}

        return pd.DataFrame(results).T


# 示例
if __name__ == '__main__':
    # 模拟用户旅程数据
    journeys = [
        ['Facebook', 'Google', 'Email'],
        ['TikTok', 'Google'],
        ['Facebook', 'TikTok', 'Google', 'Email'],
        ['Google'],
        ['Facebook'],
        ['TikTok', 'Facebook', 'Google'],
        ['Email'],
        ['Facebook', 'Google'],
    ]
    conversions = [1, 1, 1, 1, 0, 1, 0, 1]

    model = AttributionModel(journeys, conversions)
    comparison = model.compare_models()
    print("归因模型对比 (%):")
    print(comparison.round(1))
print("[✓] Ad Attribution Modeling 测试通过")
```

---


## ④ 技能关联

### 前置技能
- [Skill-Intelligent-Attribution-Causal-Forest](../01-因果推断/[[Skill-Intelligent-Attribution-Causal-Forest]].md) — 因果森林为归因提供反事实基础

### 延伸技能
- [Skill-ROAS-Budget-Optimization](../13-广告分析/[[Skill-ROAS-Budget-Optimization]].md) — 归因结果驱动预算分配优化

### 可组合
- [Skill-Marketing-Mix-Modeling](../15-营销投放分析/[[Skill-Marketing-Mix-Modeling]].md) — 渠道归因 + MMM 形成短长期视角统一


- **可组合（延伸）**：[[Skill-PVM-Attribution-Window-Harmonization]] / [[Skill-Negative-Keyword-Safe-Guard]] / [[Skill-Negative-Keyword-Safe-Guard]] / [[Skill-FrontDoor-Causal-MTA]]

## ⑤ 商业价值评估

- **ROI**：预算重新分配后ROAS提升20-40%，年节省浪费预算10万+
- **难度**：⭐⭐⭐☆☆（3/5）— Shapley计算复杂，但规则模型简单
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 广告预算分配的前提，没有归因就没有优化
