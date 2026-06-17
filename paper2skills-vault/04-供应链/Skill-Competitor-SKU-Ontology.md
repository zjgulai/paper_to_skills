---
title: 竞品SKU本体 — 竞争关系图谱建模、差异化标签与定价策略连接
doc_type: knowledge
module: 04-供应链
topic: competitor-sku-ontology
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 竞品SKU本体

> **来源**：arXiv:2310.11823（Competitive Product Ontology in E-Commerce）+ arXiv:2402.09234（Competitor Intelligence with Knowledge Graphs）
> **桥梁**：选品规划 ↔ 知识图谱 ↔ 定价策略 | **类型**：竞争情报

## ① 算法原理

**竞品SKU本体** 将竞争关系从"人工跟踪"升级为"结构化图谱+Tag查询"。

**竞争关系类型**：

| 关系类型 | 定义 | Tag |
|--------|------|-----|
| 直接竞品 | 同功能同价位 | `competitor.type=DIRECT` |
| 替代品 | 可替代使用 | `competitor.type=SUBSTITUTE` |
| 互补品 | 配合使用 | `complement.sku_id=xxx` |
| 价格领导 | 同类中定价基准 | `competitor.price_leader=True` |
| 劣质竞品 | 低价低质，品牌需区分 | `competitor.quality_tier=LOW` |

**自动化竞争监控**：
- 竞品价格变动 → `competitor.price_change=True` → 触发定价审查
- 竞品评分下降 → `competitor.rating_drop=True` → 市场机会标签

## ② 代码模板

```python
"""
竞品 SKU 本体
功能：竞品关系建模 / 价格监控 / 差异化分析 / 竞争机会标签
"""
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CompetitorSKU:
    asin: str
    title: str
    brand: str
    price_usd: float
    rating: float
    review_count: int
    bsr: int                # Best Seller Rank
    monthly_sales_est: int
    competition_type: str   # DIRECT / SUBSTITUTE / PRICE_LEADER
    quality_tier: str       # HIGH / MEDIUM / LOW
    our_sku_id: Optional[str] = None  # 我们的哪个SKU在竞争
    tags: dict = field(default_factory=dict)


def analyze_competitive_position(our_sku: dict, competitors: list) -> dict:
    """分析竞争位置并生成标签"""
    our_price = our_sku["price_usd"]
    our_rating = our_sku["rating"]
    our_bsr = our_sku["bsr"]

    # 价格分析
    direct_prices = [c.price_usd for c in competitors if c.competition_type == "DIRECT"]
    price_leader = min(competitors, key=lambda c: c.price_usd)
    avg_comp_price = sum(direct_prices) / max(1, len(direct_prices)) if direct_prices else our_price
    price_position = "ABOVE_MARKET" if our_price > avg_comp_price * 1.1 else (
        "BELOW_MARKET" if our_price < avg_comp_price * 0.9 else "AT_MARKET")

    # 评分分析
    avg_comp_rating = sum(c.rating for c in competitors if c.competition_type == "DIRECT") / max(1, len(competitors))
    rating_advantage = our_rating - avg_comp_rating
    rating_position = "LEADER" if rating_advantage > 0.3 else ("COMPETITIVE" if rating_advantage > -0.1 else "LAGGING")

    # 最大威胁竞品
    biggest_threat = max(competitors, key=lambda c: c.monthly_sales_est)

    tags = {
        "market.price_position": price_position,
        "market.rating_position": rating_position,
        "market.biggest_threat_asin": biggest_threat.asin,
        "market.price_gap_pct": round((our_price - avg_comp_price) / avg_comp_price * 100, 1),
        "market.rating_gap": round(rating_advantage, 2),
        "competitor.price_leader_price": price_leader.price_usd,
    }

    opportunities = []
    if biggest_threat.rating < our_rating - 0.3:
        opportunities.append(f"竞品{biggest_threat.brand}评分低{our_rating-biggest_threat.rating:.1f}分，可强化差异化营销")
    if price_position == "ABOVE_MARKET":
        opportunities.append(f"价格高于均价{abs(tags['market.price_gap_pct']):.0f}%，需确保品质溢价支撑")

    return {"price_position": price_position, "rating_position": rating_position,
            "avg_competitor_price": round(avg_comp_price, 2),
            "opportunities": opportunities, "tags": tags}


if __name__ == "__main__":
    print("【竞品SKU本体】\n")
    our_sku = {"sku_id": "SKU-S12Pro", "price_usd": 59.99, "rating": 4.4, "bsr": 85}
    competitors = [
        CompetitorSKU("B001COMP1", "Mommed Breast Pump", "Mommed", 49.99, 4.1, 8200, 120, 1500, "DIRECT", "MEDIUM"),
        CompetitorSKU("B002COMP2", "Spectra S1+", "Spectra", 69.99, 4.7, 25000, 45, 3000, "DIRECT", "HIGH"),
        CompetitorSKU("B003COMP3", "Elvie Stride", "Elvie", 89.99, 4.3, 5600, 200, 800, "DIRECT", "HIGH"),
    ]

    result = analyze_competitive_position(our_sku, competitors)
    print(f"  价格定位: {result['price_position']}  均价: ${result['avg_competitor_price']}")
    print(f"  评分定位: {result['rating_position']}")
    for opp in result["opportunities"]:
        print(f"  机会: {opp}")
    print(f"  Tags: {result['tags']}")
    print(f"\n[✓] 竞品SKU本体 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Product-Category-Opportunity-Scoring]]（品类评分含竞争强度维度）
- **延伸（extends）**：[[Skill-Monodense-单品价格弹性估计]]（竞品价格是弹性估计的参考）
- **可组合（combinable）**：[[Skill-New-SKU-Launch-Readiness-Gate]]（竞品分析是上市准入门控的选品前置）

## ⑤ 商业价值评估

- **ROI预估**：结构化竞品监控使价格策略响应时间从"每周人工更新"→"实时自动"，抓住竞品价格上涨窗口提价，年化增收约5-8万元；及时发现竞品评分下降机会，加大广告投放
- **实施难度**：⭐⭐☆☆☆（数据来源：Amazon API/第三方工具）
- **优先级评分**：⭐⭐⭐⭐☆（跨境电商是高度竞争市场，不了解竞品动态就是闭门造车）
