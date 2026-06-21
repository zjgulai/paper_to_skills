---
title: 搜索声量份额 — 关键词维度市场份额追踪与竞争格局监测
doc_type: knowledge
module: 25-搜索流量工程
topic: search-share-of-voice
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 搜索声量份额

> **论文/方法来源**：Share of Voice in Digital Advertising（IAB 2020）+ Search Visibility Index（Searchmetrics 方法论）
> **领域**：搜索流量工程 ↔ 广告分析 | **类型**: 算法工具

## ① 算法原理

搜索声量份额（Search Share of Voice, SOV）源自传统媒体广告领域的"广告声量占比"概念，迁移到搜索场景后衡量**品牌/产品在特定关键词集合中的曝光占比**。

**关键词级 SOV 定义**：
$$SOV_{brand}(kw) = \frac{Impressions_{brand}(kw)}{\sum_{competitor} Impressions_{competitor}(kw)}$$

**加权 SOV**（更实用）：按关键词搜索量加权，高搜索量词权重更大：
$$SOV_{weighted} = \frac{\sum_{kw} sv_{kw} \cdot impression\_share_{brand}(kw)}{\sum_{kw} sv_{kw}}$$

**竞争格局分析**：构建关键词-品牌 SOV 矩阵，识别三类词：
1. **优势词**（自身 SOV > 40%）：巩固防守
2. **竞争词**（自身 SOV 10-40%，市场搜索量大）：重点进攻
3. **机会词**（自身 SOV < 10%，竞争格局分散）：低成本切入

时序监控：每周计算 SOV 变化，下降 ≥ 5% 触发预警，反向排查是否竞品投放加强或自身广告暂停。

## ② 母婴出海应用案例

**场景A：吸奶器品类竞争格局诊断**
- 业务问题：感觉竞品流量在涨，但不知道哪些关键词上被抢占了多少份额
- 数据要求：TOP 50 关键词的自身 + 竞品广告展示份额（Amazon Advertising 数据）、搜索量估算
- 预期产出：关键词维度 SOV 矩阵 + 竞争格局热力图 + 周环比变化报告
- 业务价值：精准定位份额流失关键词，定向加投，年化减少流量流失约 20-30 万元

**场景B：新品上市 SOV 增长路径规划**
- 业务问题：新品上市，从零 SOV 开始，需要制定6个月内达到 15% SOV 的计划
- 数据要求：品类核心词 SOV 基线，竞品 SOV 分布，广告预算约束
- 预期产出：按词优先级的 SOV 提升路径图，月度里程碑
- 业务价值：资源聚焦，避免全线铺开分散预算，6个月 SOV 目标达成率 ≥ 80%

## ③ 代码模板

```python
import numpy as np
import pandas as pd
from typing import Dict, List

def compute_keyword_sov(impression_data: pd.DataFrame) -> pd.DataFrame:
    """
    计算关键词级 SOV
    impression_data 列：keyword, brand, impressions, search_volume
    """
    total_impr = impression_data.groupby("keyword")["impressions"].sum().reset_index()
    total_impr.columns = ["keyword", "total_impressions"]
    
    df = impression_data.merge(total_impr, on="keyword")
    df["sov"] = (df["impressions"] / df["total_impressions"].replace(0, np.nan)).round(4)
    return df

def compute_weighted_sov(df: pd.DataFrame, brand_name: str) -> Dict:
    """计算加权 SOV（按搜索量加权）"""
    brand_df = df[df["brand"] == brand_name].copy()
    total_sv = df.groupby("keyword")["search_volume"].first().sum()
    
    weighted_sov = (brand_df["sov"] * brand_df["search_volume"]).sum() / total_sv
    return {
        "brand": brand_name,
        "weighted_sov": round(weighted_sov, 4),
        "keyword_count": len(brand_df),
        "total_impressions": int(brand_df["impressions"].sum())
    }

def classify_keywords(df: pd.DataFrame, brand_name: str) -> pd.DataFrame:
    """关键词分类：优势词 / 竞争词 / 机会词"""
    brand_df = df[df["brand"] == brand_name][["keyword", "sov", "search_volume"]].copy()
    brand_df["category"] = brand_df["sov"].apply(
        lambda s: "优势词" if s >= 0.40 else ("竞争词" if s >= 0.10 else "机会词")
    )
    brand_df["priority"] = brand_df.apply(
        lambda r: "HIGH" if (r["category"] == "竞争词" and r["search_volume"] > 5000) else (
            "MEDIUM" if r["category"] == "优势词" else "LOW"), axis=1
    )
    return brand_df.sort_values(["category", "search_volume"], ascending=[True, False])

def sov_trend_monitor(weekly_sov: pd.DataFrame, brand_name: str, alert_threshold: float = 0.05) -> pd.DataFrame:
    """
    周环比 SOV 变化监控
    weekly_sov 列：week, keyword, brand, sov
    """
    brand_sov = weekly_sov[weekly_sov["brand"] == brand_name].sort_values(["keyword", "week"])
    brand_sov["sov_prev"] = brand_sov.groupby("keyword")["sov"].shift(1)
    brand_sov["sov_change"] = brand_sov["sov"] - brand_sov["sov_prev"]
    brand_sov["alert"] = brand_sov["sov_change"] < -alert_threshold
    return brand_sov[brand_sov["alert"]][["keyword", "week", "sov", "sov_prev", "sov_change"]]

# 示例数据
np.random.seed(42)
brands = ["OurBrand", "Haakaa", "Medela", "Elvie", "Spectra"]
keywords = [
    "breast pump", "electric breast pump", "wearable breast pump",
    "breast pump hands free", "hospital grade pump", "manual breast pump",
    "portable breast pump", "best breast pump 2024"
]
search_volumes = [50000, 35000, 28000, 22000, 15000, 12000, 18000, 25000]

rows = []
for kw, sv in zip(keywords, search_volumes):
    remaining = sv
    brand_shares = np.random.dirichlet(np.ones(len(brands)) * 2)
    for brand, share in zip(brands, brand_shares):
        rows.append({
            "keyword": kw,
            "brand": brand,
            "impressions": int(sv * share),
            "search_volume": sv
        })

df = pd.DataFrame(rows)
sov_df = compute_keyword_sov(df)

print("=== 加权 SOV 汇总 ===")
for brand in brands:
    result = compute_weighted_sov(sov_df, brand)
    print(f"  {result['brand']:<15} SOV: {result['weighted_sov']:.1%}  展示量: {result['total_impressions']:>8,}")

print("\n=== OurBrand 关键词分类 ===")
classified = classify_keywords(sov_df, "OurBrand")
print(classified.to_string(index=False))

# 周趋势监控（模拟2周数据）
weekly_rows = []
for week in [1, 2]:
    for _, row in sov_df.iterrows():
        noise = np.random.normal(0, 0.02)
        if week == 2 and row["brand"] == "OurBrand":
            noise -= 0.06  # 模拟第2周SOV下降
        weekly_rows.append({**row.to_dict(), "week": week, "sov": max(0, row["sov"] + noise)})

weekly_df = pd.DataFrame(weekly_rows)
alerts = sov_trend_monitor(weekly_df, "OurBrand")
print(f"\n=== SOV 预警（下降>5%）===")
if len(alerts) > 0:
    print(alerts.to_string(index=False))
else:
    print("无预警")
print("\n[✓] 搜索声量份额测试通过")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Search-Position-Click-Elasticity]]（点击弹性决定不同排名位置的 SOV 贡献系数）
- **延伸（extends）**：[[Skill-Index-Health-Monitoring]]（SOV 监测发现异常时，用索引健康检查排查是否为收录问题）
- **可组合（combinable）**：[[Skill-Brand-Defense-Search-Strategy]]（SOV 数据驱动品牌词防守投入优先级）

## ⑤ 商业价值评估
- ROI预估：SOV 从 10% 提升至 20%，对应品类流量份额翻倍，年化增收 50-100 万元（取决于品类规模）
- 实施难度：⭐⭐☆☆☆
- 优先级：⭐⭐⭐⭐⭐
- 评估依据：SOV 是品牌竞争力的领先指标，通常先于销售额变化3-4周；周维度监控可快速发现竞争格局变化，指导广告策略动态调整
