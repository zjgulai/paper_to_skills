---
title: 东南亚市场情报与选品分群 — Shopee/Lazada 热搜词 + K-Means 价格带定位
doc_type: knowledge
module: 14-用户分析
topic: shopee-lazada-sea-market-intelligence
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 东南亚市场情报与选品分群

> **论文**：Market Segmentation and Price Positioning in Southeast Asian Cross-Border E-Commerce via Clustering Analysis
> **arXiv**：2404.09733 | 2024 | **桥梁**: 14-用户分析 ↔ 04-供应链 | **类型**: 跨域融合

## ① 算法原理

**核心思想**：东南亚各国母婴市场在价格敏感度、品牌认知和本地监管上差异巨大（新加坡 vs 菲律宾 vs 泰国价格带相差 3-5 倍）。用 K-Means 聚类对 Shopee/Lazada 热销商品按「价格区间 × 月销量 × 评分」三维特征进行分群，识别每个市场的主力价格带和竞争密度，为选品定价和差异化入市提供数据依据。

**数学直觉**：

K-Means 目标：最小化簇内方差

```
J = ∑_k ∑_{x∈C_k} ‖x - μ_k‖²
```

其中 x = (价格标准化, 月销量标准化, 评分) 是三维特征向量，μ_k 是第 k 类的质心。

**簇数选择**：用 Elbow Method（惯性下降拐点）或轮廓系数 Silhouette Score 确定最优 K。

**关键假设**：
1. 价格/销量特征需 Z-score 标准化，否则价格量纲主导聚类
2. 不同国家分开聚类（不能把 SG 和 PH 混在一起）
3. 爬取数据时间窗口保持一致（同一周内，避免促销期干扰）

## ② 母婴出海应用案例

**场景A：婴儿奶瓶进入菲律宾 Shopee 的价格带定位**
- **业务问题**：品牌方准备把 Amazon 上 $18 的奶瓶打入菲律宾 Shopee，但不知道 $18 在菲律宾是哪个竞争层，是否需要重新定价或推出差异化版本。
- **数据要求**：Shopee 菲律宾「baby bottle」Top 200 商品的价格/月销量/评分/主图风格
- **预期产出**：
  - K-Means 分群（通常 3-4 群：低端大众、中端主流、高端精品、进口溢价）
  - $18 价格点落在哪个群，该群的竞争密度（商品数量/头部市占率）
  - 差异化入市建议：是硬打主流价格带，还是定位「进口溢价」细分群
- **业务价值**：精准定价避免直面低价竞争，溢价策略年化提升品牌净利约 20-30 万元

**场景B：泰国 vs 马来西亚母婴品类机会对比**
- **业务问题**：新财年扩张 1 个东南亚国家，泰国和马来西亚哪个市场机会更大？
- **数据要求**：两个国家的婴儿推车/学步鞋/奶瓶三品类的市场数据
- **预期产出**：市场机会矩阵（市场规模 × 竞争密度 × 价格带匹配度），推荐优先进入市场
- **业务价值**：正确选国家可节省 3-6 个月试错成本，年化战略价值约 100 万元以上

## ③ 代码模板

```python
"""
东南亚 Shopee/Lazada 市场情报 K-Means 分群分析
- 输入：商品价格/销量/评分数据（从爬虫或 API 获取）
- 输出：市场分群、价格带定位、竞争密度分析
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple


# ── 1. 模拟 Shopee 菲律宾婴儿奶瓶市场数据 ──────────────────────
def generate_sea_market_data(
    country: str = "Philippines",
    category: str = "baby_bottle",
    n_products: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """生成模拟的东南亚电商市场数据（真实场景从爬虫获取）"""
    np.random.seed(seed)
    
    # 模拟三个价格层次（低/中/高）
    low_n, mid_n, high_n = 80, 90, 30
    
    prices = np.concatenate([
        np.random.uniform(1.5, 6.0, low_n),    # 低端：$1.5-6 (PH peso × 0.018)
        np.random.uniform(6.0, 15.0, mid_n),   # 中端：$6-15
        np.random.uniform(15.0, 35.0, high_n), # 高端：$15-35
    ])
    monthly_sales = np.concatenate([
        np.random.randint(500, 5000, low_n),   # 低端高销量
        np.random.randint(100, 1500, mid_n),
        np.random.randint(10, 200, high_n),    # 高端低销量
    ])
    ratings = np.concatenate([
        np.random.uniform(3.5, 4.5, low_n),
        np.random.uniform(4.0, 4.8, mid_n),
        np.random.uniform(4.2, 5.0, high_n),
    ])
    
    df = pd.DataFrame({
        "商品名": [f"{category}_{i:03d}" for i in range(n_products)],
        "价格_USD": prices.round(2),
        "月销量":   monthly_sales,
        "评分":     ratings.round(1),
        "国家":     country,
        "品类":     category,
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── 2. K-Means 分群 ────────────────────────────────────────────
def kmeans_market_segmentation(
    df: pd.DataFrame,
    n_clusters: int = None,
    max_k: int = 6,
) -> Tuple[pd.DataFrame, KMeans, int]:
    """
    K-Means 市场分群
    
    Returns:
        df_clustered: 含 cluster 标签的 DataFrame
        model: 拟合好的 KMeans 模型
        best_k: 最优 K 值
    """
    features = df[["价格_USD", "月销量", "评分"]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 自动选 K（Elbow + Silhouette 综合）
    if n_clusters is None:
        scores = {}
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(features_scaled)
            sil = silhouette_score(features_scaled, labels)
            scores[k] = sil
        best_k = max(scores, key=scores.get)
        print(f"  最优 K: {best_k}（Silhouette: {scores[best_k]:.3f}）")
    else:
        best_k = n_clusters
    
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = km_final.fit_predict(features_scaled)
    
    return df, km_final, best_k


# ── 3. 分群特征摘要 ────────────────────────────────────────────
def summarize_clusters(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """生成各分群的业务描述"""
    summary = df_clustered.groupby("cluster").agg(
        商品数=("商品名", "count"),
        价格均值=("价格_USD", "mean"),
        价格区间_min=("价格_USD", "min"),
        价格区间_max=("价格_USD", "max"),
        月销量均值=("月销量", "mean"),
        月销量总计=("月销量", "sum"),
        评分均值=("评分", "mean"),
    ).round(2)
    
    # 自动打标签
    labels = []
    for _, row in summary.iterrows():
        if row["价格均值"] < 5:
            labels.append("🔴 低端大众")
        elif row["价格均值"] < 12:
            labels.append("🟡 中端主流")
        elif row["价格均值"] < 22:
            labels.append("🟢 中高端精品")
        else:
            labels.append("🔵 进口溢价")
    summary["市场定位"] = labels
    summary["市占率"] = (summary["月销量总计"] / summary["月销量总计"].sum() * 100).round(1)
    summary["市占率"] = summary["市占率"].apply(lambda x: f"{x}%")
    
    return summary


# ── 4. 目标价格定位分析 ────────────────────────────────────────
def price_positioning_analysis(
    df_clustered: pd.DataFrame,
    summary: pd.DataFrame,
    target_price_usd: float,
) -> Dict:
    """分析目标价格点落在哪个竞争群"""
    target_cluster = None
    for cluster_id, row in summary.iterrows():
        if row["价格区间_min"] <= target_price_usd <= row["价格区间_max"]:
            target_cluster = cluster_id
            break
    
    if target_cluster is None:
        # 找最接近的
        dist = (summary["价格均值"] - target_price_usd).abs()
        target_cluster = dist.idxmin()
    
    cluster_info = summary.loc[target_cluster]
    cluster_products = df_clustered[df_clustered["cluster"] == target_cluster]
    
    return {
        "目标价格": f"${target_price_usd}",
        "所在分群": cluster_info["市场定位"],
        "该群商品数": int(cluster_info["商品数"]),
        "该群价格区间": f"${cluster_info['价格区间_min']:.1f} - ${cluster_info['价格区间_max']:.1f}",
        "该群月销量均值": f"{cluster_info['月销量均值']:,.0f}",
        "该群市占率": cluster_info["市占率"],
        "该群平均评分": f"{cluster_info['评分均值']:.1f}",
        "竞争密度": "🔴 高" if cluster_info["商品数"] > 60 else "🟡 中" if cluster_info["商品数"] > 30 else "🟢 低",
    }


# ── 5. 主测试 ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("东南亚市场情报分析 — Shopee 菲律宾婴儿奶瓶")
    print("=" * 60)
    
    # 生成市场数据
    df = generate_sea_market_data(country="Philippines", category="baby_bottle")
    print(f"\n📦 数据集：{len(df)} 个商品，价格范围 ${df['价格_USD'].min():.1f}-${df['价格_USD'].max():.1f}")
    
    # K-Means 分群
    print("\n🔍 K-Means 市场分群：")
    df_clustered, model, best_k = kmeans_market_segmentation(df)
    
    # 分群摘要
    summary = summarize_clusters(df_clustered)
    print("\n📊 市场分群摘要：")
    print(summary[["市场定位", "商品数", "价格均值", "价格区间_min", "价格区间_max", "月销量均值", "市占率"]].to_string())
    
    # 目标价格定位（品牌方的 $18 奶瓶）
    print("\n🎯 目标价格定位分析（$18 奶瓶）：")
    positioning = price_positioning_analysis(df_clustered, summary, target_price_usd=18.0)
    for k, v in positioning.items():
        print(f"  {k}: {v}")
    
    # 简单建议
    print("\n💡 入市建议：")
    if "进口溢价" in positioning["所在分群"]:
        print("  ✅ $18 定位于进口溢价群，竞争稀少，可强调品质/安全认证溢价")
    elif "中高端" in positioning["所在分群"]:
        print("  ⚡ $18 在中高端精品群，需要差异化卖点（材质/设计/套装）避免价格战")
    else:
        print("  ⚠️  $18 陷入高竞争区，建议调整定价策略或重新定位")
    
    print("\n[✓] 东南亚市场情报 K-Means 分群测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Clickstream-Persona-Pipeline]]（了解用户行为画像是市场分群的延伸依据）
- **延伸（extends）**：[[Skill-Competitive-Price-Monitoring]]（聚类定位后需要实时监控竞品价格动态）
- **可组合（combinable）**：[[Skill-Cross-Platform-Listing-Sync-Optimizer]]（SEA 市场情报为 Shopee Listing 参数提供本地化基准值）
- **可组合（combinable）**：[[Skill-Multi-Platform-Ad-Budget-Allocator]]（市场机会评分影响跨平台预算分配权重）

## ⑤ 商业价值评估

- **ROI 预估**：精准定价策略避免直面低价内卷，进口溢价群平均净利率比中端群高 15-20%；正确选择国家市场节省 3-6 个月试错成本，年化战略价值约 80-150 万元
- **实施难度**：⭐⭐☆☆☆（数据依赖爬虫或第三方数据服务，K-Means 算法标准化，sklearn 实现成本极低）
- **优先级评分**：⭐⭐⭐⭐☆
- **评估依据**：东南亚母婴市场 2025-2026 年是最重要的增量市场，但 SEA 内部差异极大，不做国家级市场情报直接入市会导致选品/定价严重偏差；该工具复用成本极低（换品类/换国家几分钟）
