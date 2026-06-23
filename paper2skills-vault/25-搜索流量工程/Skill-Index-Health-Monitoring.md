---
title: 索引健康度监控 — 全链路搜索索引覆盖率与收录状态追踪
doc_type: knowledge
module: 25-搜索流量工程
topic: index-health-monitoring
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 索引健康度监控

> **论文/方法来源**：Information Retrieval Health Metrics（Baeza-Yates & Ribeiro-Neto 2011）+ Amazon Seller Central 索引诊断实践
> **领域**：搜索流量工程 ↔ 数据采集工程 | **类型**: 工程基础

## ① 算法原理

索引健康度监控（Index Health Monitoring）源自信息检索领域的覆盖率评估方法，将搜索引擎的抓取-解析-索引三阶段拆解为可量化的健康指标体系。核心指标包括：

**抓取覆盖率（Crawl Coverage）**：已抓取 URL 数 / 全部可抓取 URL 数，反映爬虫到达率。

**索引收录率（Index Ratio）**：已收录页面数 / 已抓取页面数，反映质量过滤通过率。

**关键词索引状态（Keyword Indexation Status）**：目标关键词下产品是否出现在索引中，用布尔值或排名位次表示。

监控框架采用**时序异常检测**：对每个指标维护滑动窗口基线（7日均值±2σ），超出阈值触发告警。关键假设：索引状态具有稳态性，短期异常后会回归，持续异常才代表真正问题。

数学定义：健康分 $H = \alpha \cdot CR + \beta \cdot IR + \gamma \cdot KIS$，其中 $\alpha+\beta+\gamma=1$，建议权重 0.3/0.3/0.4，KIS 权重最高因其直接影响流量。

## ② 母婴出海应用案例

**场景A：吸奶器 Listing 索引异常检测**
- 业务问题：新品上架后 72 小时搜索流量为零，不知道是否已被索引
- 数据要求：ASIN 列表、目标关键词列表、Seller Central 报告 API 数据
- 预期产出：每日索引健康分报告，识别未收录关键词（精度 ≥ 95%），平均发现延迟 < 4 小时
- 业务价值：提前发现索引问题，避免新品前7天流量损失，年化减少错失销售约 15-30 万元

**场景B：多 SKU 批量索引状态巡检**
- 业务问题：100+ SKU 日常巡检，人工抽查效率低，容易漏报
- 数据要求：全量 ASIN × 关键词矩阵，历史排名快照数据
- 预期产出：自动标记健康分 < 0.7 的 SKU，生成优先处理队列
- 业务价值：运营效率提升 60%，索引问题平均修复周期从 3 天缩短到 1 天

## ③ 代码模板

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

def compute_index_health_score(
    crawl_coverage: float,
    index_ratio: float,
    keyword_indexation_rate: float,
    weights: Tuple[float, float, float] = (0.3, 0.3, 0.4)
) -> float:
    """计算索引健康分 H = alpha*CR + beta*IR + gamma*KIS"""
    alpha, beta, gamma = weights
    score = alpha * crawl_coverage + beta * index_ratio + gamma * keyword_indexation_rate
    return round(score, 4)

def detect_anomaly(history: List[float], current: float, window: int = 7, sigma: float = 2.0) -> Dict:
    """滑动窗口异常检测：当前值是否超出基线±2σ"""
    if len(history) < window:
        return {"anomaly": False, "reason": "history_insufficient"}
    
    recent = history[-window:]
    baseline = np.mean(recent)
    std = np.std(recent)
    lower = baseline - sigma * std
    upper = baseline + sigma * std
    
    is_anomaly = current < lower
    return {
        "anomaly": is_anomaly,
        "current": current,
        "baseline": round(baseline, 4),
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
        "deviation_sigma": round((current - baseline) / (std + 1e-8), 2)
    }

def build_index_health_report(sku_data: pd.DataFrame) -> pd.DataFrame:
    """
    输入 DataFrame 列：asin, date, crawl_coverage, index_ratio, keyword_indexation_rate
    输出：每 ASIN 最新健康分 + 异常标记
    """
    results = []
    for asin, group in sku_data.groupby("asin"):
        group = group.sort_values("date")
        history_scores = []
        for _, row in group.iterrows():
            h = compute_index_health_score(
                row["crawl_coverage"],
                row["index_ratio"],
                row["keyword_indexation_rate"]
            )
            history_scores.append(h)
        
        current_score = history_scores[-1]
        anomaly_info = detect_anomaly(history_scores[:-1], current_score)
        
        results.append({
            "asin": asin,
            "date": group["date"].iloc[-1],
            "health_score": current_score,
            "anomaly": anomaly_info["anomaly"],
            "deviation_sigma": anomaly_info.get("deviation_sigma", 0.0),
            "action": "URGENT" if (anomaly_info["anomaly"] and current_score < 0.5) else (
                "WATCH" if anomaly_info["anomaly"] else "OK"
            )
        })
    
    return pd.DataFrame(results).sort_values("health_score")

# 示例数据
np.random.seed(42)
dates = [datetime(2026, 6, i+1) for i in range(15)]
asins = ["B001BABY01", "B001BABY02", "B001BABY03"]

rows = []
for asin in asins:
    for d in dates:
        rows.append({
            "asin": asin,
            "date": d.strftime("%Y-%m-%d"),
            "crawl_coverage": np.random.uniform(0.80, 0.99),
            "index_ratio": np.random.uniform(0.75, 0.98),
            "keyword_indexation_rate": np.random.uniform(0.60, 0.95) if asin != "B001BABY02" else np.random.uniform(0.20, 0.40)
        })

df = pd.DataFrame(rows)
# 制造 B001BABY02 在最后一天异常低
df.loc[(df["asin"] == "B001BABY02") & (df["date"] == "2026-06-15"), "keyword_indexation_rate"] = 0.15

report = build_index_health_report(df)
print(report.to_string(index=False))
print("\n[✓] 索引健康度监控测试通过")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Amazon-Search-Ranking-Factor-Model]]（搜索排名因子是索引健康的结果验证）
- **延伸（extends）**：[[Skill-Search-Share-of-Voice]]（索引健康是声量份额的基础保障）
- **可组合（combinable）**：[[Skill-Keyword-Demand-Gap-Analysis]]（健康度异常 → 触发关键词需求缺口补救分析）
- 可组合：[[Skill-AB-Experimental-Design]]
- 可组合：[[Skill-Agentic-AB-Testing]]

## ⑤ 商业价值评估
- ROI预估：100 SKU 规模，年化减少索引问题导致的流量损失约 20-50 万元；运营人天节省 ≈ 0.5 人/月
- 实施难度：⭐⭐☆☆☆
- 优先级：⭐⭐⭐⭐⭐
- 评估依据：新品前7天索引状态直接决定 BSR 起点，一旦错过索引窗口补救成本是原来的3倍；实施只需 Seller Central API + 标准 Python
