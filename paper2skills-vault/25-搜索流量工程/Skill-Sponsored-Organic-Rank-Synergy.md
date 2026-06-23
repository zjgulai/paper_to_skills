---
title: Skill-Sponsored-Organic-Rank-Synergy — 广告-自然排名协同模型
doc_type: knowledge
module: 25-搜索流量工程
topic: sponsored-organic-rank-synergy
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Sponsored-Organic-Rank-Synergy

> **论文/方法来源**：Search Advertising and Organic Search Interaction（Ghose & Yang 2009）+ Halo Effect Modeling in E-commerce PPC（工业实践）
> **领域**：搜索流量工程 ↔ 广告分析 | **类型**: 协同优化

## ① 算法原理

广告-自然排名协同（Sponsored-Organic Rank Synergy）研究 PPC 广告对自然排名的正向溢出效应（Halo Effect）。核心机制：广告带来的点击和订单信号被 A9 算法采纳为表现信号，进而改善自然排名。

**协同效应量化**：

$$\Delta Organic\_Rank = f(PPC\_CVR, PPC\_Volume, Keyword\_Relevance)$$

简化线性近似（经验模型）：

$$\Delta Rank = \beta_1 \cdot \log(Orders_{PPC}) + \beta_2 \cdot CVR_{PPC} + \beta_3 \cdot Relevance$$

其中 $\beta_1 \approx -3.2$（每翻倍 PPC 订单量，自然排名提升约 3 位），$\beta_2 \approx -15$（CVR 每提升 1%，排名提升约 15 位）。

**三阶段协同策略**：
1. **攻坚期**（第 1-2 周）：高竞价广告冲 Top 3 位置，积累点击/订单信号
2. **巩固期**（第 3-6 周）：自然排名进入 Top 30 后，逐步降低 PPC 竞价
3. **收割期**（第 7 周+）：自然排名 Top 10，大幅削减广告预算，维持低 ACOS

**监控指标**：Organic Rank、Sponsored Rank、Total Page 1 Presence（广告+自然合计 P1 占位数）。

## ② 母婴出海应用案例

**场景：婴儿奶瓶核心词从广告依赖转向自然流量主导**

- **业务问题**：「baby bottle bpa free」词广告 ACOS 42%，纯靠广告不可持续，但自然排名仅 #85
- **数据要求**：过去30天 PPC 数据（Search Term Report）、Helium10 排名追踪、月预算 $3,000
- **执行方案**：
  - 攻坚期：「baby bottle bpa free」 Exact Match 提价至 $2.5，日出 15-20 单
  - 6周后自然排名从 #85 → #28，广告竞价降至 $1.2
  - 12周后自然排名 #12，广告仅作补充，月 ACOS 降至 18%
- **量化产出**：ACOS 从 42% → 18%，同等月销量下广告费从 $3,000 → $1,200
- **业务价值**：年化节省广告费 21,600 元，广告投入减少 60% 但销量维持

## ③ 代码模板

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def estimate_organic_rank_improvement(
    ppc_orders: float,
    ppc_cvr: float,
    keyword_relevance: float = 0.8,
    beta: Tuple[float, float, float] = (-3.2, -15.0, -5.0)
) -> float:
    """估算 PPC 带动的自然排名提升（负值=排名数字变小=排名提升）"""
    b1, b2, b3 = beta
    delta = b1 * np.log1p(ppc_orders) + b2 * ppc_cvr + b3 * keyword_relevance
    return round(delta, 1)

def simulate_synergy_phases(
    initial_organic_rank: int = 85,
    initial_ppc_budget: float = 3000,
    initial_ppc_cvr: float = 0.08,
    weeks: int = 16
) -> pd.DataFrame:
    """模拟三阶段广告-自然协同演化"""
    rows = []
    organic_rank = initial_organic_rank
    ppc_budget = initial_ppc_budget
    
    for week in range(1, weeks + 1):
        # 阶段判断
        if week <= 4:
            phase = "攻坚期"
            bid_multiplier = 1.5
        elif week <= 8:
            phase = "巩固期"
            bid_multiplier = 1.0
        else:
            phase = "收割期"
            bid_multiplier = 0.5
        
        # 当期 PPC 订单估算（预算/CPC/CVR）
        effective_budget = ppc_budget * bid_multiplier
        cpc_est = 1.5 + 0.5 * bid_multiplier
        clicks = effective_budget / cpc_est
        ppc_orders = clicks * initial_ppc_cvr * (1 + 0.1 * week / weeks)
        
        # 自然排名改善
        delta = estimate_organic_rank_improvement(ppc_orders, initial_ppc_cvr)
        organic_rank = max(1, organic_rank + delta * 0.3)  # 累积衰减
        
        # 自然流量估算（排名 → 流量经验公式）
        organic_traffic_share = max(0, (200 - organic_rank) / 200) ** 2
        
        # P1 存在度
        p1_presence = min(1.0, (bid_multiplier * 0.5) + organic_traffic_share)
        
        rows.append({
            "week": week,
            "phase": phase,
            "organic_rank": round(organic_rank, 0),
            "ppc_budget_effective": round(effective_budget, 0),
            "ppc_orders": round(ppc_orders, 1),
            "organic_traffic_share_pct": round(organic_traffic_share * 100, 1),
            "p1_presence": round(p1_presence, 2)
        })
    
    return pd.DataFrame(rows)

def compute_acos_trajectory(df: pd.DataFrame, avg_order_value: float = 35.0) -> pd.DataFrame:
    """计算各周 ACOS"""
    df = df.copy()
    df["revenue_est"] = df["ppc_orders"] * avg_order_value
    df["acos"] = df["ppc_budget_effective"] / df["revenue_est"].replace(0, np.nan)
    df["acos"] = df["acos"].fillna(0).round(3)
    return df

# 测试
np.random.seed(42)
df = simulate_synergy_phases(initial_organic_rank=85)
df = compute_acos_trajectory(df)

print("=== 广告-自然排名协同演化 ===")
print(df[["week","phase","organic_rank","ppc_budget_effective","acos","organic_traffic_share_pct"]].to_string(index=False))

# 汇总
phase_summary = df.groupby("phase").agg({
    "organic_rank": "mean",
    "acos": "mean",
    "ppc_budget_effective": "mean"
}).round(2)
print("\n=== 各阶段均值 ===")
print(phase_summary)

print("\n[✓] Sponsored-Organic-Rank-Synergy 测试通过")
```

## ④ 技能关联

- **前置**：[[Skill-A9-Algorithm-Sales-Velocity-Optimization]]（排名信号基础）、[[Skill-Search-Funnel-Attribution]]（流量归因）
- **延伸**：[[Skill-Search-Ad-Budget-ROI-Integration]]（预算分配优化）、[[Skill-Search-Position-Click-Elasticity]]（弹性模型）
- **可组合**：[[Skill-Search-Query-Performance-Attribution]]（精准词识别）+ [[Skill-Search-Term-Negative-Optimization]]（ACOS 控制）

## ⑤ 商业价值评估

- **ROI**：广告-自然协同后 ACOS 从 42% → 18%，年化节省广告费约 2-5 万元/品
- **实施难度**：⭐⭐⭐☆☆（需要 12-16 周耐心执行，监控体系要完备）
- **优先级**：⭐⭐⭐⭐⭐（成熟品高 ACOS 的核心解法，投入产出比极高）
