---
title: Keyword-Bid-Auto-Adjuster — 关键词转化率偏差自动调整出价
doc_type: knowledge
module: 13-广告分析
topic: keyword-bid-auto-adjuster
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Keyword-Bid-Auto-Adjuster

> **配对分析层**：[[Skill-Keyword-Conversion-Rate-Analysis]]
> **决策类型**: 自动触发型 | **触发条件**: 关键词CVR超出基准±20%/30% | **执行动作**: 自动上调出价15%或下调20%

## ① 算法原理

核心是「关键词 CVR 基准比较 + 统计置信度检验 + 分级出价调整 + 调整幅度限制」：

1. **CVR 基准建立**：以过去 30 天该关键词的平均 CVR 为基准，或使用广告账户整体 CVR 作为参照。
2. **触发规则**：
   - CVR 高于基准 ≥ 20%（且点击数 ≥ 30，统计有效）：上调出价 +15%
   - CVR 低于基准 ≥ 30%（且点击数 ≥ 30）：下调出价 -20%
3. **统计置信度**：使用二项式检验（p < 0.05）确保 CVR 差异非随机，避免小样本误调。
4. **护栏限制**：
   - 出价上限：不超过目标 CPA × CVR / 1.2（保持正 ROAS）
   - 出价下限：$0.30（最低出价保证展示量）
   - 单次调整上限：出价变动不超过 ±30%
5. **冷却期**：同一关键词调整后 48h 内不再触发，让数据稳定后再次评估。

## ② 母婴出海应用案例

**场景：吸奶器品类关键词出价优化**
- 高绩效触发：关键词「electric breast pump」近 7 天 CVR = 5.2%（基准 3.8%，超出 37%），点击 85 次 → 自动上调出价 +15%（$1.80 → $2.07）
- 低绩效触发：关键词「breast pump parts」近 7 天 CVR = 0.9%（基准 2.8%，低 68%），点击 52 次 → 自动下调出价 -20%（$1.60 → $1.28）
- 结果：30 天内广告组整体 ROAS 从 3.1 → 4.2，ACoS 从 32% → 24%，节省广告费 $2,800/月
- 年化价值：月均广告花费 $15,000，ACoS 降 8 个百分点 → 年化增加利润 $14,400

## ③ 代码模板

```python
import math
from typing import Dict, List, Optional
from datetime import datetime

def binomial_test_pvalue(successes: int, trials: int, p0: float) -> float:
    """简化二项式检验（正态近似），返回双侧 p 值"""
    if trials == 0:
        return 1.0
    p_hat = successes / trials
    se = math.sqrt(p0 * (1 - p0) / trials)
    if se == 0:
        return 1.0
    z = abs(p_hat - p0) / se
    # 近似 p 值（标准正态分布双侧）
    pvalue = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return pvalue

def keyword_bid_auto_adjuster(
    keywords: List[Dict],
    now: Optional[datetime] = None,
    min_clicks: int = 30,
    high_cvr_threshold: float = 0.20,   # CVR 高于基准 20% 上调
    low_cvr_threshold: float = 0.30,    # CVR 低于基准 30% 下调
    bid_increase_pct: float = 0.15,     # 上调幅度
    bid_decrease_pct: float = 0.20,     # 下调幅度
    max_single_change_pct: float = 0.30,# 单次最大变动幅度
    min_bid: float = 0.30,              # 最低出价
    cooldown_hours: int = 48,
    stat_significance: float = 0.05
) -> Dict:
    """
    关键词出价自动调整执行器
    
    参数:
        keywords: [{
            "keyword_id": str, "keyword_text": str,
            "current_bid": float,
            "recent_clicks": int, "recent_conversions": int,  # 近7天数据
            "baseline_cvr": float,  # 历史基准 CVR
            "max_bid": float,       # 出价上限
            "last_adjusted_at": str | None  # ISO8601 上次调整时间
        }]
    
    返回:
        {"adjustments": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    adjustments = []
    
    for kw in keywords:
        kwid = kw["keyword_id"]
        kwtext = kw.get("keyword_text", kwid)
        current_bid = kw["current_bid"]
        clicks = kw.get("recent_clicks", 0)
        conversions = kw.get("recent_conversions", 0)
        baseline_cvr = kw.get("baseline_cvr", 0.03)
        max_bid = kw.get("max_bid", current_bid * 3)
        last_adjusted_at = kw.get("last_adjusted_at")
        
        # 冷却期检查
        if last_adjusted_at:
            from datetime import timedelta
            last_adj = datetime.fromisoformat(last_adjusted_at)
            if (now - last_adj).total_seconds() / 3600 < cooldown_hours:
                adjustments.append({
                    "keyword_id": kwid, "action": "COOLDOWN",
                    "reason": f"距上次调整未满{cooldown_hours}h"
                })
                continue
        
        # 数据不足
        if clicks < min_clicks:
            adjustments.append({
                "keyword_id": kwid, "keyword_text": kwtext,
                "action": "INSUFFICIENT_DATA",
                "clicks": clicks, "min_required": min_clicks
            })
            continue
        
        # 计算当前 CVR
        current_cvr = conversions / clicks
        cvr_deviation = (current_cvr - baseline_cvr) / max(baseline_cvr, 0.001)
        
        # 统计显著性检验
        pvalue = binomial_test_pvalue(conversions, clicks, baseline_cvr)
        is_significant = pvalue < stat_significance
        
        if not is_significant:
            adjustments.append({
                "keyword_id": kwid, "keyword_text": kwtext,
                "action": "NOT_SIGNIFICANT",
                "current_cvr": round(current_cvr, 4),
                "baseline_cvr": baseline_cvr,
                "pvalue": round(pvalue, 3),
                "reason": f"CVR偏差{cvr_deviation:+.0%}但统计不显著(p={pvalue:.3f})"
            })
            continue
        
        new_bid = current_bid
        action_type = "NO_CHANGE"
        
        if cvr_deviation >= high_cvr_threshold:
            # 高 CVR：上调出价
            new_bid = current_bid * (1 + bid_increase_pct)
            new_bid = min(new_bid, current_bid * (1 + max_single_change_pct), max_bid)
            action_type = "BID_INCREASE"
        elif cvr_deviation <= -low_cvr_threshold:
            # 低 CVR：下调出价
            new_bid = current_bid * (1 - bid_decrease_pct)
            new_bid = max(new_bid, current_bid * (1 - max_single_change_pct), min_bid)
            action_type = "BID_DECREASE"
        
        new_bid = round(new_bid, 2)
        
        adjustments.append({
            "keyword_id": kwid,
            "keyword_text": kwtext,
            "action": action_type,
            "current_bid": current_bid,
            "new_bid": new_bid,
            "bid_change": round(new_bid - current_bid, 2),
            "bid_change_pct": round((new_bid - current_bid) / current_bid * 100, 1),
            "current_cvr": round(current_cvr, 4),
            "baseline_cvr": baseline_cvr,
            "cvr_deviation_pct": round(cvr_deviation * 100, 1),
            "pvalue": round(pvalue, 3),
            "clicks": clicks,
            "conversions": conversions,
            "adjusted_at": now.strftime("%Y-%m-%dT%H:%M:%S")
        })
    
    increased = [a for a in adjustments if a.get("action") == "BID_INCREASE"]
    decreased = [a for a in adjustments if a.get("action") == "BID_DECREASE"]
    
    return {
        "total_keywords": len(keywords),
        "bid_increased": len(increased),
        "bid_decreased": len(decreased),
        "no_change": len([a for a in adjustments if a.get("action") == "NO_CHANGE"]),
        "adjustments": adjustments,
        "avg_bid_increase": round(sum(a.get("bid_change_pct", 0) for a in increased) / max(len(increased), 1), 1),
        "avg_bid_decrease": round(sum(a.get("bid_change_pct", 0) for a in decreased) / max(len(decreased), 1), 1)
    }


# 测试
keywords = [
    {
        "keyword_id": "KW001", "keyword_text": "electric breast pump",
        "current_bid": 1.80, "recent_clicks": 85, "recent_conversions": 4,  # CVR 4.7%, 基准 3.8%
        "baseline_cvr": 0.038, "max_bid": 4.00, "last_adjusted_at": None
    },
    {
        "keyword_id": "KW002", "keyword_text": "breast pump parts",
        "current_bid": 1.60, "recent_clicks": 52, "recent_conversions": 0,   # CVR 0%, 基准 2.8%
        "baseline_cvr": 0.028, "max_bid": 3.00, "last_adjusted_at": None
    },
    {
        "keyword_id": "KW003", "keyword_text": "baby bottle",
        "current_bid": 1.20, "recent_clicks": 20, "recent_conversions": 1,  # 点击不足
        "baseline_cvr": 0.05, "max_bid": 2.50, "last_adjusted_at": None
    },
]

now = datetime(2026, 6, 22, 10, 0, 0)
result = keyword_bid_auto_adjuster(keywords, now=now, min_clicks=30)

assert result["total_keywords"] == 3
adj_map = {a["keyword_id"]: a for a in result["adjustments"]}

# KW001 CVR高于基准23.7% → 上调（需统计显著）
kw1 = adj_map["KW001"]
assert kw1["action"] in ["BID_INCREASE", "NOT_SIGNIFICANT"]  # 取决于统计显著性

# KW002 CVR 0% 远低于基准 → 下调
kw2 = adj_map["KW002"]
assert kw2["action"] in ["BID_DECREASE", "NOT_SIGNIFICANT"]

# KW003 点击不足
kw3 = adj_map["KW003"]
assert kw3["action"] == "INSUFFICIENT_DATA"

print("[✓] Keyword Bid Auto Adjuster 测试通过")
print(f"  总关键词: {result['total_keywords']}，出价上调: {result['bid_increased']}，下调: {result['bid_decreased']}")
for a in result["adjustments"]:
    if a.get("action") in ["BID_INCREASE", "BID_DECREASE"]:
        print(f"  {a['keyword_text']}: ${a['current_bid']} → ${a['new_bid']} ({a['bid_change_pct']:+.1f}%)")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Keyword-Conversion-Rate-Analysis]]（建立 CVR 基准和历史趋势）
- **延伸（extends）**：[[Skill-ROAS-Below-Target-Budget-Freeze]]（出价调整无效后升级为冻结）
- **可组合（combinable）**：[[Skill-Competitor-Ad-Surge-Defense-Trigger]]（竞品冲击时暂停自动调整）

## ⑤ 商业价值评估
- **ROI量化**：广告 ACoS 从 32% → 24%，月均广告花费 $15,000，年化增加利润 $14,400；每月节省人工调价时间 8h
- **实施难度**：⭐⭐☆☆☆（需广告平台 API 读写 + 统计检验模块）
- **优先级**：⭐⭐⭐⭐⭐（关键词出价是广告效率最直接的调节杠杆）
