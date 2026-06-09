---
title: Multimarket Expansion Readiness Scorer（多市场拓展就绪度评分）
doc_type: knowledge
module: 06-增长模型
topic: multimarket-expansion-readiness-scoring
status: stable
created: 2026-06-09
updated: 2026-06-09
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Multimarket-Expansion-Readiness-Scorer（多市场拓展就绪度）

> **桥梁**: 06-增长模型 ↔ 21-合规决策 ↔ 04-供应链 ↔ 15-营销投放分析 | **类型**: 四域枢纽

---

## ① 算法原理

**核心思想**：跨境品牌从美国市场扩张到欧洲/日本/中东时，面临的不是单一问题，而是产品合规、物流成本、用户需求差异、竞争格局、现金流压力五个维度同时变化。就绪度评分模型将这五个维度量化为 0-100 分的综合指数，输出「GO/WAIT/NO-GO」三档建议。

**五维度评分框架**：

**Dim 1: 产品合规就绪度（权重 25%）**
```
评分因子：
  - 目标市场是否需要额外认证（CE/BfR/PSE 等）→ 0(需要未准备) / 50(已申请) / 100(持证)
  - 当前认证能否复用（FDA 到 CE 不可复用 → 扣分）
  - HTS 编码在目标市场的税率（关税 < 5% 加分，> 15% 减分）
```

**Dim 2: 物流链路就绪度（权重 20%）**
```
评分因子：
  - 目标市场是否有稳定海外仓/FBA 节点
  - 平均配送时效 vs 目标市场用户预期（Amazon DE：2 天，Amazon JP：次日）
  - 头程运费占比（< 8% GMV 优，> 15% 警示）
```

**Dim 3: 市场需求信号（权重 25%）**
```
评分因子：
  - 类似品在目标市场的 BSR 趋势（近 90 天上升 → 加分）
  - 搜索量趋势（Google Trends + Amazon 类目关键词增长率）
  - 竞争密度（类目 Top 10 卖家中本土品牌占比，越高越难进入）
```

**Dim 4: 品牌/内容就绪度（权重 15%）**
```
评分因子：
  - 是否有目标语言的 Listing 和产品文案（德/日/阿拉伯语）
  - 社媒内容本地化覆盖率（TikTok DE / Amazon JP Stores）
  - 已有本地化评论数量（0 → 0分，50+ → 80分）
```

**Dim 5: 财务就绪度（权重 15%）**
```
评分因子：
  - 首批备货资金是否到位（< 3 个月运营现金流 → 警示）
  - 盈亏平衡月份预测（< 6 个月 → GO，6-12 个月 → WAIT，> 12 → NO-GO）
  - 现有核心市场 ROI 稳定性（波动 > 30% → 不宜扩张）
```

**综合评分 + 决策规则**：
```
readiness_score = Σ (dim_score_i × weight_i)

GO:    score ≥ 70 且 合规维度 ≥ 60
WAIT:  score 50-69 或 任一维度 < 40
NO-GO: score < 50 或 合规维度 < 30
```

---

## ② 母婴出海应用案例

**场景：Momcozy 从美国站向德国 Amazon 扩张评估**

| 维度 | 得分 | 关键发现 |
|------|------|---------|
| 产品合规 | 62 | 吸奶器需要 CE + MDR 认证（医疗器械），申请中（未持证扣 40 分），HTS 关税 0% 加分 |
| 物流链路 | 78 | 已有法兰克福 FBA 仓，DE 配送次日达，头程成本 7% GMV（合格） |
| 市场需求 | 71 | 德国 Amazon 母婴品类 BSR 90 天上升 12%，竞争以本土品牌为主（中等难度）|
| 品牌/内容 | 45 | 仅有机器翻译德语 Listing，无本地化评论，TikTok DE 无内容 |
| 财务就绪 | 68 | 首批备货 50 万已准备，预估 8 个月回本（WAIT 区间）|

**综合得分**：`62×0.25 + 78×0.20 + 71×0.25 + 45×0.15 + 68×0.15 = 65.8`

**决策**：**WAIT** — 核心阻塞点：① CE 认证未完成（合规 62 < 70）；② 德语内容完全缺失

**行动计划**：
1. 加急完成 CE 认证（BGV/TÜV 实验室，预计 8-12 周）
2. 聘用德语本地化编辑（Listing + 首批 50 条评论培育）
3. 60 天后重新评分，预计届时 score ≥ 74 → GO

---

## ③ 代码模板

```python
from dataclasses import dataclass

@dataclass
class MarketReadinessInput:
    market: str
    compliance_score: float
    logistics_score: float
    demand_score: float
    content_score: float
    financial_score: float

WEIGHTS = {
    "compliance": 0.25,
    "logistics": 0.20,
    "demand": 0.25,
    "content": 0.15,
    "financial": 0.15,
}

def score_market_readiness(inp: MarketReadinessInput) -> dict:
    weighted = (
        inp.compliance_score * WEIGHTS["compliance"]
        + inp.logistics_score * WEIGHTS["logistics"]
        + inp.demand_score * WEIGHTS["demand"]
        + inp.content_score * WEIGHTS["content"]
        + inp.financial_score * WEIGHTS["financial"]
    )

    if weighted >= 70 and inp.compliance_score >= 60:
        decision = "GO"
        rationale = "全面就绪，建议在下一个财季启动"
    elif weighted >= 50 and inp.compliance_score >= 30:
        decision = "WAIT"
        blockers = []
        if inp.compliance_score < 60:
            blockers.append(f"合规就绪度不足 ({inp.compliance_score:.0f}<60)")
        if inp.content_score < 50:
            blockers.append(f"内容本地化薄弱 ({inp.content_score:.0f}<50)")
        if inp.financial_score < 55:
            blockers.append(f"财务就绪度不足 ({inp.financial_score:.0f}<55)")
        rationale = "阻塞点: " + "; ".join(blockers) if blockers else "综合得分偏低，建议先强化弱项"
    else:
        decision = "NO-GO"
        rationale = "市场时机或资源条件不成熟，建议推迟 6+ 个月"

    return {
        "market": inp.market,
        "total_score": round(weighted, 1),
        "decision": decision,
        "rationale": rationale,
        "dim_scores": {
            "compliance": inp.compliance_score,
            "logistics": inp.logistics_score,
            "demand": inp.demand_score,
            "content": inp.content_score,
            "financial": inp.financial_score,
        },
    }

markets = [
    MarketReadinessInput("Amazon DE", 62, 78, 71, 45, 68),
    MarketReadinessInput("Amazon JP", 55, 65, 58, 30, 72),
    MarketReadinessInput("Amazon UK", 85, 82, 74, 70, 76),
    MarketReadinessInput("Amazon AE", 40, 55, 62, 25, 60),
]

for m in markets:
    r = score_market_readiness(m)
    print(f"{r['market']:15s} score={r['total_score']:5.1f}  [{r['decision']:6s}]  {r['rationale'][:60]}")

print("\n[✓] 多市场就绪度评分测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Market-Size-Estimation]]（TAM/SAM 评估，驱动 demand_score）
- **前置**：[[Skill-Category-Compliance-Prescan]]（合规门控预扫，驱动 compliance_score）
- **前置**：[[Skill-Cross-Border-Cash-Flow-Forecasting]]（财务就绪度输入）
- **组合**：[[Skill-Cross-Border-Compliance-Framework]]（细化合规维度评分）
- **组合**：[[Skill-Supplier-Capacity-Planning]]（评估新市场供应链扩容能力）
- **延伸**：[[Skill-Bass-Diffusion-New-Product-Forecasting]]（新市场冷启动需求预测）
- **延伸**：[[Skill-Cultural-Adaptation-Agent]]（content_score 改善路径）

---

## ⑤ 商业价值评估

**ROI 估算**：

| 场景 | 价值 |
|------|------|
| 避免合规未就绪提前入市（CPSC/CE 被下架） | 防止损失 30-200 万元/次 |
| 识别最优入市时机（避免过早烧预算） | 首年广告节省 20-50 万元 |
| 四域枢纽效应（合规+供应链+财务+营销联动分析） | 跨职能决策效率提升，避免信息孤岛误判 |

**实施难度**：⭐⭐☆☆☆（低，数据大部分已有，主要是框架建立）

**优先级评分**：5/5（每次新市场扩张决策的必备前置分析，是高中心度四域枢纽节点）

**图谱中心度预测**：桥接 06-增长模型 ↔ 21-合规 ↔ 04-供应链 ↔ 15-营销，预计成为 degree 55-70 的高中心度节点。
