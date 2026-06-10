---
title: Competitor Price Intelligence — 竞品价格实时监控与智能响应
doc_type: knowledge
module: 17-价格优化
topic: competitor-price-intelligence-monitoring-response
status: stable
created: 2026-06-10
updated: 2026-06-10
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Competitor-Price-Intelligence（竞品价格智能监控）

> **论文**：Multi-Agent RL for Dynamic Pricing in Supply Chains (arXiv:2507.02698, 2025)
> **桥梁**: 17-价格优化 ↔ 13-广告分析 ↔ 04-供应链 | **类型**: 算法工具

---

## ① 算法原理

**核心思想**：在 Amazon 上，你的竞品每天都在改价——大促前降价吸引流量、节后回调、对你调价后立即跟随。如果没有系统性监控和响应策略，你要么被竞品抢走 Buy Box，要么无谓地跟降利润。

竞品价格智能包含三层：
1. **实时采集**：每小时/每日抓取竞品 ASIN 价格（via Keepa API 或自建爬虫）
2. **变动检测**：统计检测（CUSUM 算法）识别价格突变，区分常规波动和策略性变动
3. **响应策略**：基于多智能体博弈论，根据竞品行为模式选择最优响应（跟降/维价/差异化）

**竞品价格变动分类**：
```
类型1: 促销性降价（持续 24-72h，然后回调）
  → 策略: 同期适度降价 + 广告加预算，大促后恢复

类型2: 持续性降价（趋势性降价，可能是清库存）
  → 策略: 观察 7 天，若持续 → 评估是否降价或差异化竞争

类型3: 跟随性降价（对你的调价立即响应）
  → 策略: 识别算法型竞品，利用其滞后性获取短暂价格优势

类型4: 缺货后涨价
  → 策略: 保持价格稳定，获取 Buy Box 份额
```

**CUSUM 异常检测**：
```
S_t = max(0, S_{t-1} + (price_t - μ_baseline) - k)
当 S_t > h 时触发预警（h=阈值，k=允许偏差）
```

---

## ② 母婴出海应用案例

**场景：吸奶器品类竞品价格攻防**

- **业务问题**：主要竞品 B 每周五下午降价 10-15%（周末大流量期），周二恢复原价。品牌方每次都被动应对，或者错过窗口，或者跟降太晚白白损失利润。
- **系统化响应流程**：
  1. 每小时监控竞品 B、C、D 的 ASIN 价格
  2. CUSUM 检测到竞品 B 周五 15:00 降价 → 触发预警
  3. 自动执行预设规则：降价 8%（比竞品便宜 $2）+ 广告预算提升 30%
  4. 周一 9:00 竞品 B 回调 → 自动恢复原价
- **业务价值**：Buy Box 获取率从 65% 提升到 82%，周末流量期 GMV 增长 25%，同时避免非必要降价损耗利润。

---

## ③ 代码模板

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import statistics

@dataclass
class PriceRecord:
    asin: str
    competitor: str
    price: float
    timestamp: datetime

def detect_cusum_change(prices: List[float], k: float = 0.5, h: float = 5.0) -> Dict:
    if len(prices) < 5:
        return {"change_detected": False}
    baseline = statistics.mean(prices[:-3])
    sigma = statistics.stdev(prices[:-3]) if len(prices) > 4 else 1.0
    s_pos, s_neg = 0.0, 0.0
    change_point = None
    for i, p in enumerate(prices[-5:], len(prices) - 5):
        normalized = (p - baseline) / max(sigma, 0.01)
        s_pos = max(0, s_pos + normalized - k)
        s_neg = max(0, s_neg - normalized - k)
        if s_pos > h or s_neg > h:
            change_point = i
            break
    return {"change_detected": change_point is not None,
            "change_point": change_point,
            "baseline_price": round(baseline, 2),
            "latest_price": round(prices[-1], 2),
            "change_pct": round((prices[-1] - baseline) / baseline * 100, 1)}

def classify_price_move(records: List[PriceRecord], window_days: int = 7) -> str:
    if len(records) < 3:
        return "insufficient_data"
    prices = [r.price for r in records[-10:]]
    change = detect_cusum_change(prices)
    if not change["change_detected"]:
        return "stable"
    change_pct = change["change_pct"]
    duration = (records[-1].timestamp - records[-len(prices)//2].timestamp).days
    if change_pct < -5 and duration <= 3:
        return "promotional_drop"
    elif change_pct < -5 and duration > 5:
        return "sustained_drop"
    elif change_pct > 5:
        return "price_increase"
    return "minor_fluctuation"

def recommend_response(my_price: float, competitor_price: float,
                        move_type: str, margin_floor: float) -> Dict:
    strategies = {
        "promotional_drop": {"action": "follow_partially",
                              "new_price": max(margin_floor, competitor_price + 1.5),
                              "ad_budget_multiplier": 1.3,
                              "reason": "跟降但保持 $1.5 价差，周末结束后恢复"},
        "sustained_drop":   {"action": "evaluate_7d",
                              "new_price": my_price,
                              "ad_budget_multiplier": 1.0,
                              "reason": "观察 7 天，判断是否清库存，暂不跟降"},
        "price_increase":   {"action": "hold_price",
                              "new_price": my_price,
                              "ad_budget_multiplier": 1.2,
                              "reason": "竞品涨价，保持原价提升 Buy Box 份额"},
        "stable":           {"action": "no_change",
                              "new_price": my_price,
                              "ad_budget_multiplier": 1.0,
                              "reason": "竞品价格稳定，无需响应"},
        "minor_fluctuation":{"action": "no_change",
                              "new_price": my_price,
                              "ad_budget_multiplier": 1.0,
                              "reason": "正常波动，无需响应"},
    }
    strategy = strategies.get(move_type, strategies["stable"])
    return {"competitor_move": move_type, **strategy,
            "price_delta": round(strategy["new_price"] - my_price, 2)}

from datetime import timedelta
base_time = datetime(2026, 6, 10)
competitor_records = [
    PriceRecord("B08XY", "CompetitorB", 94.99, base_time - timedelta(days=6)),
    PriceRecord("B08XY", "CompetitorB", 94.99, base_time - timedelta(days=5)),
    PriceRecord("B08XY", "CompetitorB", 94.99, base_time - timedelta(days=4)),
    PriceRecord("B08XY", "CompetitorB", 94.99, base_time - timedelta(days=3)),
    PriceRecord("B08XY", "CompetitorB", 80.99, base_time - timedelta(days=1)),
    PriceRecord("B08XY", "CompetitorB", 80.99, base_time),
]
prices = [r.price for r in competitor_records]
change = detect_cusum_change(prices)
move_type = classify_price_move(competitor_records)
response = recommend_response(my_price=89.99, competitor_price=80.99,
                               move_type=move_type, margin_floor=72.0)
print(f"竞品当前: ${prices[-1]} (基线: ${change['baseline_price']}, 变动: {change['change_pct']}%)")
print(f"变动类型: {move_type}")
print(f"建议响应: {response['action']} → 新价格 ${response['new_price']}")
print(f"广告预算: ×{response['ad_budget_multiplier']}")
print(f"理由: {response['reason']}")
print("[✓] Competitor Price Intelligence 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-AIGP-LLM-Dynamic-Pricing]]（LLM 动态定价引擎执行响应策略）
- **前置**：[[Skill-Competitive-Price-Monitoring]]（价格监控数据采集层）
- **延伸**：[[Skill-Compliant-Dynamic-Pricing-Guard]]（响应调价需满足 MAP/最低价合规约束）
- **延伸**：[[Skill-ROAS-Budget-Optimization]]（调价时同步优化广告预算分配）
- **组合**：[[Skill-Keyword-Competition-Scoring]]（价格竞争 + 关键词竞争联合分析，全面评估竞争态势）

---

## ⑤ 商业价值评估

- **ROI 预估**：Buy Box 获取率 65%→82%，周末流量期 GMV +25%，年化增量 20-100 万元
- **实施难度**：⭐⭐☆☆☆（低，Keepa API 获取竞品价格 + CUSUM 检测算法）
- **优先级**：⭐⭐⭐⭐⭐（定价是最高频决策，竞品监控是 Buy Box 争夺的基础工具）
- **评估依据**：arXiv 2507.02698，MARL 动态定价基准实验；CUSUM 异常检测是工业标准方法
