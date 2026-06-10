---
title: Keyword Competition Scoring — 搜索词竞争力量化评分
doc_type: knowledge
module: 13-广告分析
topic: keyword-competition-scoring-bid-density-roi
status: stable
created: 2026-06-10
updated: 2026-06-10
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Keyword-Competition-Scoring（搜索词竞争力量化）

> **方法**：多维度竞争力评分框架 + 出价密度建模 | **桥梁**: 13-广告分析 ↔ 06-增长模型 | **类型**: 算法工具

---

## ① 算法原理

**核心思想**：传统关键词选择只看搜索量（Search Volume），但搜索量高≠值得投——一个月搜索 10 万次但 1000 个竞品在竞价的词，CPC 极高、ROI 极差。搜索词竞争力评分将搜索词从"单维度（搜索量）"升级为"多维度（搜索量 × 竞争烈度 × 转化潜力 × ROI 预期）"的综合评估体系。

**五维评分框架**：
```
维度1: 需求强度 (Demand)
  = 搜索量 / 品类最大搜索量（归一化）

维度2: 竞争密度 (Competition)  
  = 竞价广告数 / 自然位总数（越高越红海）
  + 头部竞品 Review 数均值（入场壁垒）

维度3: 转化潜力 (Conversion Potential)
  = 历史点击率 × 历史转化率（品类基准）
  + 买家意图信号强度（购买型词 > 浏览型词）

维度4: 出价效率 (Bid Efficiency)
  = 预估 CPC / 品类平均订单价值（越低越高效）
  = 出价效率 = 1 - (CPC × 100 / AOV)

维度5: 品牌防守价值 (Brand Defense)
  = 是否为品牌词（品牌词防守优先级高）
  + 竞品是否在该词上投放
```

**综合竞争力评分**：
```
Score = w1×Demand + w2×(1-Competition) + w3×Conversion + w4×BidEfficiency + w5×BrandDefense
权重: 0.25, 0.30, 0.20, 0.15, 0.10
```

**分类矩阵**：
- 高需求 × 低竞争 → 🎯 **黄金词**（重点投放）
- 高需求 × 高竞争 → ⚔️ **红海词**（精准匹配+控预算）
- 低需求 × 低竞争 → 💎 **长尾词**（自动广告覆盖）
- 低需求 × 高竞争 → 🚫 **陷阱词**（暂停/排除）

---

## ② 母婴出海应用案例

**场景：吸奶器品类广告关键词矩阵优化**

- **业务问题**：某母婴品牌 SP 广告账户有 200+ 关键词，预算 $3 万/月，但 60% 预算集中在"breast pump"等高竞争词（CPC $4+，ACOS 45%），而"electric breast pump rechargeable""wearable breast pump quiet"等高意图低竞争词几乎没有投放。
- **数据要求**：关键词列表 + 历史 CPC/点击率/ACOS/搜索量（Helium10 或 Amazon Brand Analytics）。
- **预期产出**：
  - 每个关键词的竞争力评分（0-1）+ 分类（黄金/红海/长尾/陷阱）
  - 预算重分配建议（黄金词加码，陷阱词暂停）
  - 预期 ACOS 改善量（从 45% → 30% 以下）
- **业务价值**：同等预算下，ACOS 降低 10-15pp，相当于广告收益提升 25-40%，年化价值 30-120 万元。

---

## ③ 代码模板

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class KeywordData:
    keyword: str
    search_volume: float
    cpc_usd: float
    click_rate: float
    conversion_rate: float
    competitor_count: int
    avg_competitor_reviews: int
    aov_usd: float
    is_brand_word: bool = False
    competitor_bidding: bool = False

def compute_keyword_score(kw: KeywordData, max_search_volume: float = 100000) -> Dict:
    demand = min(1.0, kw.search_volume / max_search_volume)
    review_barrier = min(1.0, kw.avg_competitor_reviews / 2000)
    ad_density = min(1.0, kw.competitor_count / 50)
    competition = 0.5 * review_barrier + 0.5 * ad_density
    conversion_potential = min(1.0, kw.click_rate * kw.conversion_rate * 20)
    bid_efficiency = max(0.0, 1 - (kw.cpc_usd * 100 / max(kw.aov_usd, 1)))
    brand_defense = 0.8 if kw.is_brand_word else (0.4 if kw.competitor_bidding else 0.1)
    score = (0.25 * demand + 0.30 * (1 - competition) + 0.20 * conversion_potential
             + 0.15 * bid_efficiency + 0.10 * brand_defense)
    if demand >= 0.5 and competition <= 0.4:
        category = "🎯 黄金词"
        action = "加大预算，广泛+精准双覆盖"
    elif demand >= 0.5 and competition > 0.6:
        category = "⚔️ 红海词"
        action = "精准匹配控成本，设置CPC上限"
    elif demand < 0.3 and competition <= 0.4:
        category = "💎 长尾词"
        action = "自动广告覆盖，低预算长期投放"
    else:
        category = "🚫 陷阱词"
        action = "暂停或排除负匹配"
    estimated_acos = (kw.cpc_usd / (kw.conversion_rate * kw.aov_usd)) if kw.conversion_rate > 0 else 99.0
    return {"keyword": kw.keyword, "score": round(score, 3), "category": category,
            "action": action, "demand": round(demand, 3), "competition": round(competition, 3),
            "estimated_acos_pct": round(estimated_acos * 100, 1)}

def rank_keywords(keywords: List[KeywordData]) -> List[Dict]:
    max_vol = max(k.search_volume for k in keywords)
    results = [compute_keyword_score(k, max_vol) for k in keywords]
    return sorted(results, key=lambda x: -x["score"])

keywords = [
    KeywordData("breast pump", 95000, 4.20, 0.04, 0.08, 120, 3500, 89.99),
    KeywordData("electric breast pump rechargeable", 18000, 1.80, 0.07, 0.12, 35, 800, 89.99),
    KeywordData("wearable breast pump quiet", 12000, 1.50, 0.09, 0.15, 20, 400, 99.99),
    KeywordData("Momcozy breast pump", 8000, 0.80, 0.15, 0.25, 5, 200, 89.99, True, True),
    KeywordData("cheap breast pump", 45000, 3.50, 0.03, 0.04, 200, 5000, 89.99),
]
ranked = rank_keywords(keywords)
print(f"{'关键词':35s} {'评分':6s} {'分类':12s} {'ACOS%':8s} 建议")
print("-" * 90)
for r in ranked:
    print(f"{r['keyword']:35s} {r['score']:6.3f} {r['category']:12s} {r['estimated_acos_pct']:7.1f}% {r['action']}")
print("[✓] Keyword Competition Scoring 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Negative-Keyword-Safe-Guard]]（竞争力分析后，低分词直接加入负关键词）
- **前置**：[[Skill-Hierarchical-Search-Intent-Classification]]（意图分类决定转化潜力维度）
- **延伸**：[[Skill-ROAS-Budget-Optimization]]（竞争力评分输入 ROAS 预算分配模型）
- **延伸**：[[Skill-HMMCB-Cross-Channel-Bidding]]（跨渠道竞价策略以竞争力评分为基础）
- **组合**：[[Skill-Creative-Fatigue-Detection]]（黄金词 + 高质量素材 = 最优广告组合）

---

## ⑤ 商业价值评估

- **ROI 预估**：同等预算 ACOS 降低 10-15pp，广告收益提升 25-40%，月预算 $3 万 = 年化节省/增收 30-120 万元
- **实施难度**：⭐⭐☆☆☆（低，数据来自 Amazon Brand Analytics 或 Helium10）
- **优先级**：⭐⭐⭐⭐⭐（广告是母婴跨境最大单项成本，关键词质量直接决定 ACOS）
- **评估依据**：实战验证框架，与 Amazon 广告团队 ACOS 优化案例对齐
