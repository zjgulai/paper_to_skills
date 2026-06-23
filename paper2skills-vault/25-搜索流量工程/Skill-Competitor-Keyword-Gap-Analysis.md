---
title: Skill-Competitor-Keyword-Gap-Analysis — 竞品关键词缺口分析
doc_type: knowledge
module: 25-搜索流量工程
topic: competitor-keyword-gap-analysis
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Competitor-Keyword-Gap-Analysis

> **论文/方法来源**：Competitive Analysis via Set Difference Methods（IR Foundations）+ Reverse ASIN Keyword Mining（工程实践）
> **领域**：搜索流量工程 ↔ 竞品分析 | **类型**: 关键词策略

## ① 算法原理

竞品关键词缺口分析（Competitor Keyword Gap Analysis）通过集合论运算，找出"竞品有排名而自己没有"的高价值关键词。

**集合定义**：
- $KW_{self}$：自己 ASIN 已有自然排名（Top 100）的关键词集合
- $KW_{comp_i}$：竞品 ASIN $i$ 已有自然排名的关键词集合
- **关键词缺口（Gap）**：$Gap = (\bigcup_{i} KW_{comp_i}) \setminus KW_{self}$

**缺口词优先级评分**：

$$Gap\_Score_k = Coverage_k \times Volume_k \times (1 - Difficulty_k)$$

其中 $Coverage_k$ = 有该词排名的竞品数/总竞品数，$Volume_k$ = 月搜索量（归一化），$Difficulty_k$ = 竞争难度（0-1）。

**实施策略**：
1. 用 Helium10/DataDive 反查 3-5 个主竞品 ASIN 的关键词
2. 导出为集合，计算差集
3. 对 Gap 词按上述评分排序，取 Top 30 词
4. 分批投入 Broad Match PPC 测试真实 CVR，确认后转 Exact

## ② 母婴出海应用案例

**场景：婴儿睡袋新品关键词蓝海挖掘**

- **业务问题**：自家婴儿睡袋 ASIN 月搜索词覆盖量 80 个，竞品 A 覆盖 350 个、竞品 B 覆盖 290 个，不知道哪些词被竞品独吃
- **数据要求**：3-5 个竞品 ASIN、Helium10 Cerebro 导出关键词列表（含排名/搜索量）
- **执行方案**：
  - 提取 3 个竞品各自 Top 150 关键词
  - 计算 Gap 词（≈180 个未覆盖词）
  - 按 Gap Score 排序，优先测试搜索量 >1,000/月的 Gap 词（约 35 个）
  - 对月搜索量 >5,000 的 Top Gap 词单独建 Campaign
- **量化产出**：新增 35 个高价值 Gap 词的覆盖，3 个月后自然排名词库扩展到 200 个
- **业务价值**：新增自然流量 30-50%，年化搜索流量价值增加约 12-20 万元

## ③ 代码模板

```python
import numpy as np
import pandas as pd
from typing import List, Dict, Set

def compute_keyword_gap(
    self_keywords: Set[str],
    competitor_keywords: List[Set[str]]
) -> Set[str]:
    """计算关键词缺口：竞品有而自己没有"""
    all_comp_kw = set()
    for kw_set in competitor_keywords:
        all_comp_kw.update(kw_set)
    return all_comp_kw - self_keywords

def score_gap_keywords(
    gap_keywords: Set[str],
    competitor_keywords: List[Set[str]],
    keyword_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    对缺口词评分
    keyword_meta 需包含列: keyword, monthly_volume, difficulty (0-1)
    """
    n_comps = len(competitor_keywords)
    rows = []
    
    meta_dict = keyword_meta.set_index("keyword").to_dict("index")
    
    for kw in gap_keywords:
        coverage = sum(1 for kw_set in competitor_keywords if kw in kw_set) / n_comps
        meta = meta_dict.get(kw, {"monthly_volume": 0, "difficulty": 0.8})
        volume_norm = min(meta["monthly_volume"] / 10000, 1.0)
        difficulty = meta["difficulty"]
        
        score = coverage * volume_norm * (1 - difficulty)
        rows.append({
            "keyword": kw,
            "coverage": round(coverage, 2),
            "monthly_volume": meta["monthly_volume"],
            "difficulty": difficulty,
            "gap_score": round(score, 4)
        })
    
    return pd.DataFrame(rows).sort_values("gap_score", ascending=False)

def classify_gap_actions(df: pd.DataFrame) -> pd.DataFrame:
    """将缺口词按优先级分类行动"""
    df = df.copy()
    def action(row):
        if row["gap_score"] > 0.05 and row["monthly_volume"] >= 5000:
            return "PRIORITY_CAMPAIGN"   # 独立 Campaign
        elif row["gap_score"] > 0.02 and row["monthly_volume"] >= 1000:
            return "BROAD_MATCH_TEST"    # Broad 测试
        elif row["monthly_volume"] >= 500:
            return "AUTO_CAMPAIGN"       # 放入 Auto
        else:
            return "MONITOR"            # 观察
    df["action"] = df.apply(action, axis=1)
    return df

def analyze_gap_summary(scored_df: pd.DataFrame) -> Dict:
    """生成缺口分析汇总报告"""
    priority = scored_df[scored_df["action"] == "PRIORITY_CAMPAIGN"]
    broad = scored_df[scored_df["action"] == "BROAD_MATCH_TEST"]
    
    return {
        "total_gap_keywords": len(scored_df),
        "priority_campaign_count": len(priority),
        "broad_match_test_count": len(broad),
        "top5_gap_keywords": scored_df.head(5)["keyword"].tolist(),
        "total_gap_volume": int(scored_df["monthly_volume"].sum()),
        "avg_difficulty": round(scored_df["difficulty"].mean(), 2)
    }

# 测试
np.random.seed(42)

self_kw = {f"kw_{i}" for i in range(80)}
comp1_kw = {f"kw_{i}" for i in range(200)}
comp2_kw = {f"kw_{i}" for i in range(50, 280)}
comp3_kw = {f"kw_{i}" for i in range(100, 320)}

gap = compute_keyword_gap(self_kw, [comp1_kw, comp2_kw, comp3_kw])
print(f"缺口词总数: {len(gap)}")

# 构造元数据
all_gap_kws = list(gap)
meta = pd.DataFrame({
    "keyword": all_gap_kws,
    "monthly_volume": np.random.randint(100, 15000, len(all_gap_kws)),
    "difficulty": np.random.uniform(0.1, 0.9, len(all_gap_kws))
})

scored = score_gap_keywords(gap, [comp1_kw, comp2_kw, comp3_kw], meta)
scored = classify_gap_actions(scored)

print("\n=== Top 10 缺口词 ===")
print(scored.head(10).to_string(index=False))

summary = analyze_gap_summary(scored)
print("\n=== 汇总报告 ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n[✓] Competitor-Keyword-Gap-Analysis 测试通过")
```

## ④ 技能关联

- **前置**：[[Skill-Long-Tail-Keyword-Mining]]（长尾词基础）、[[Skill-Search-Share-of-Voice]]（市场份额监控）
- **延伸**：[[Skill-Sponsored-Organic-Rank-Synergy]]（从 Gap 词到排名）、[[Skill-Seasonal-Keyword-Rotation-Strategy]]（季节性 Gap）
- **可组合**：[[Skill-Search-Query-Performance-Attribution]]（Gap 词 CVR 验证）+ [[Skill-Search-Term-Negative-Optimization]]（清洗低价值词）

## ⑤ 商业价值评估

- **ROI**：发现 30-50 个高价值 Gap 词 → 3 个月后自然词库扩展 50%，年化自然流量增量 12-20 万元
- **实施难度**：⭐⭐☆☆☆（依赖第三方工具导出数据，逻辑简单）
- **优先级**：⭐⭐⭐⭐☆（竞品分析必做动作，新品和成熟品均适用）
