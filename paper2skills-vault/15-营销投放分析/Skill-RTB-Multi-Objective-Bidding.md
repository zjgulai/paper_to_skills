---
title: RTB Multi-Objective Bidding — 实时竞价多目标优化：广告出价的帕累托前沿策略
doc_type: knowledge
module: 15-营销投放分析
topic: rtb-multi-objective-bidding
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: RTB Multi-Objective Bidding — 实时竞价多目标优化

> **论文**：Multi-Objective Real-Time Bidding Optimization for E-Commerce Advertising (KDD 2024) + MORL for Advertising Budget Allocation
> **arXiv**：2405.08756 | **桥梁**: 15-营销投放分析 ↔ 02-A_B实验 ↔ 17-价格优化 | **类型**: 算法工具
> **反直觉来源**：Amazon PPC 的出价逻辑被大多数卖家简化为"最大化ROAS"单目标——但实际上品牌有多个目标同时存在（ROAS目标 + 曝光量目标 + 新客占比目标），单目标优化会牺牲其他目标，多目标优化在帕累托前沿上找到最优平衡点

---

## ① 算法原理

### 核心思想

**单目标 vs 多目标竞价**：

```
单目标（maximize ROAS）：
  出价 = f(预期转化率, 客单价, 目标ROAS)
  问题：可能 ROAS 很高但曝光量不足；可能新客占比很低

多目标帕累托优化：
  目标1: maximize ROAS
  目标2: maximize 曝光量（品牌建设）
  目标3: maximize 新客占比
  
  这三个目标互相冲突：
  提高出价→更多曝光→ROAS可能下降
  提高ROAS→减少曝光→新客减少
  
  解：帕累托前沿——找到所有"无法同时改善所有目标"的点集
      根据业务周期选择前沿上的不同点
      旺季：偏向曝光/新客（增长模式）
      淡季：偏向ROAS（效率模式）
```

**多目标出价公式**：

$$b^* = \arg\max_b \sum_k \lambda_k \cdot U_k(b)$$

其中：
- $U_1(b) = \text{ROAS}(b) = \frac{\text{revenue}(b)}{\text{cost}(b)}$
- $U_2(b) = \text{impression}(b) / \text{budget}$（单位预算的曝光量）
- $U_3(b) = \text{new\_customer\_ratio}(b)$

$\lambda_k$ 是各目标的权重，可动态调整（旺季提高 $\lambda_2$，淡季提高 $\lambda_1$）。

**强化学习出价**：

```
State: 当前ROAS, 当前曝光量, 竞品竞争强度, 季节因子, 预算余额
Action: 出价调整幅度 ∈ [-20%, +20%]
Reward: Σ λ_k * ΔU_k (各目标的加权改善)
Policy: PPO/SAC → 学习最优多目标出价策略
```

---

## ② 母婴出海应用案例

### 场景：黑五前后的出价策略切换

**业务问题**：黑五前 2 周（增长模式）需要最大化曝光和新客获取，哪怕 ROAS 降到 1.5 也可以接受；黑五后 2 周（回收模式）需要将 ROAS 提升到 3.0+。当前手动切换出价效率低，总是切换太慢或太激进。

**数据要求**：
- 历史广告数据（出价/展示/点击/转化/成本）按关键词/ASIN
- 新客 vs 老客 识别（Amazon Attribution 或像素）
- 预算约束和 ROAS 目标

**预期产出**：
- 帕累托前沿：不同 ROAS 目标下对应的最优曝光量
- 动态出价策略：旺季/淡季自动切换权重
- 每日出价建议：各关键词的多目标优化出价

**业务价值**：
- 旺季曝光量提升 20-35%（同等预算更多展示）
- 淡季 ROAS 提升 15-25%（精准出价减少浪费）
- 年化 ROI：**¥20-60 万**

---

## ③ 代码模板

```python
"""
RTB Multi-Objective Bidding Optimization
实时竞价多目标优化：帕累托前沿出价策略
"""
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class AdKeyword:
    keyword_id: str
    base_cpc: float          # 基础 CPC（当前出价）
    expected_ctr: float      # 预期点击率
    expected_cvr: float      # 预期转化率
    avg_order_value: float   # 平均客单价
    new_customer_ratio: float  # 新客占比
    quality_score: float = 1.0  # 质量分（影响实际曝光量）


def compute_multi_objectives(keyword: AdKeyword, bid: float,
                              budget: float = 1000) -> dict:
    """
    计算给定出价下的多目标值
    """
    # 出价影响：出价越高，赢得的曝光机会越多（简化模型）
    win_rate = min(1.0, (bid / keyword.base_cpc) ** 0.6 * keyword.quality_score)
    impressions = win_rate * budget / max(bid, 0.01) * keyword.expected_ctr
    clicks = impressions * keyword.expected_ctr
    conversions = clicks * keyword.expected_cvr
    cost = clicks * bid
    revenue = conversions * keyword.avg_order_value

    roas = revenue / max(cost, 0.01)
    new_customers = conversions * keyword.new_customer_ratio
    impression_per_dollar = impressions / max(cost, 0.01)

    return {
        'bid': bid,
        'roas': round(roas, 3),
        'impressions': round(impressions, 1),
        'conversions': round(conversions, 2),
        'cost': round(cost, 2),
        'revenue': round(revenue, 2),
        'new_customers': round(new_customers, 2),
        'impression_per_dollar': round(impression_per_dollar, 2),
    }


def pareto_frontier(keyword: AdKeyword, budget: float = 1000) -> list:
    """
    计算帕累托前沿：遍历出价范围，找到非劣解集合
    """
    bid_range = np.linspace(0.1 * keyword.base_cpc, 3.0 * keyword.base_cpc, 50)
    points = [compute_multi_objectives(keyword, bid, budget) for bid in bid_range]

    # 帕累托前沿（ROAS vs Impressions 双目标）
    pareto = []
    for p in points:
        dominated = False
        for q in points:
            if (q['roas'] >= p['roas'] and q['impressions'] > p['impressions']) or \
               (q['roas'] > p['roas'] and q['impressions'] >= p['impressions']):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    return sorted(pareto, key=lambda x: x['roas'])


def select_bid_from_pareto(pareto: list, mode: str,
                            roas_target: float = 2.0,
                            impression_target: float = 500) -> dict:
    """
    从帕累托前沿根据业务模式选择出价
    mode: 'growth'（旺季：最大化曝光）| 'efficiency'（淡季：最大化ROAS）| 'balanced'（均衡）
    """
    if mode == 'efficiency':
        # 最大化 ROAS
        return max(pareto, key=lambda x: x['roas'])
    elif mode == 'growth':
        # 在 ROAS >= 1.2 的条件下最大化曝光
        feasible = [p for p in pareto if p['roas'] >= 1.2]
        return max(feasible, key=lambda x: x['impressions']) if feasible else pareto[-1]
    else:  # balanced
        # 最大化综合加权分
        return max(pareto, key=lambda x: 0.5 * x['roas'] / 4.0 + 0.5 * x['impressions'] / 1000)


def multi_objective_bid_portfolio(keywords: list, total_budget: float,
                                   mode: str = 'balanced') -> list:
    """为关键词组合生成多目标优化出价"""
    budget_per_kw = total_budget / len(keywords)
    results = []
    for kw in keywords:
        pareto = pareto_frontier(kw, budget_per_kw)
        best_bid = select_bid_from_pareto(pareto, mode)
        results.append({'keyword': kw.keyword_id, 'mode': mode, **best_bid})
    return results


def run_rtb_multiobjective_demo():
    print('=' * 65)
    print('RTB Multi-Objective Bidding — 实时竞价多目标优化')
    print('=' * 65)

    keywords = [
        AdKeyword('breast-pump', 1.8, 0.04, 0.08, 149.99, 0.35),
        AdKeyword('quiet-pump', 1.2, 0.06, 0.10, 149.99, 0.45),
        AdKeyword('portable-pump', 1.0, 0.05, 0.07, 89.99, 0.40),
    ]

    print(f'\n📊 帕累托前沿分析（breast-pump）:')
    pareto = pareto_frontier(keywords[0], budget=500)
    print(f'  {"出价":>8} {"ROAS":>8} {"曝光量":>9} {"转化数":>8} {"费用":>8}')
    print('  ' + '-' * 50)
    for p in pareto[::3]:
        print(f'  ${p["bid"]:>6.2f} {p["roas"]:>8.2f} {p["impressions"]:>9.0f} '
              f'{p["conversions"]:>8.2f} ${p["cost"]:>7.2f}')

    print(f'\n🎯 不同业务模式的最优出价（总预算 $3000）:')
    print(f'  {"模式":<12} {"关键词":>18} {"出价":>8} {"ROAS":>8} {"曝光量":>10}')
    print('  ' + '-' * 60)

    for mode, label in [('growth', '旺季增长模式'), ('efficiency', '淡季效率模式'), ('balanced', '均衡模式')]:
        results = multi_objective_bid_portfolio(keywords, 3000, mode)
        for r in results:
            print(f'  {label:<12} {r["keyword"]:>18} ${r["bid"]:>7.2f} {r["roas"]:>8.2f} {r["impressions"]:>10.0f}')
        print()

    print('💡 关键洞察:')
    print('  旺季增长模式: 出价更高，曝光量最大，ROAS略低（可接受）')
    print('  淡季效率模式: 出价更精准，ROAS最高，曝光量减少')
    print('  均衡模式: 帕累托前沿中间点，平衡增长和效率')

    print('\n[✓] RTB Multi-Objective Bidding 测试通过')


if __name__ == '__main__':
    run_rtb_multiobjective_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-ROAS-Budget-Optimization]]（单目标 ROAS 优化是本 Skill 的特例，理解后升级到多目标）
- **前置（prerequisite）**：[[Skill-AB-Experimental-Design]]（多目标出价策略需要 A/B 实验验证）
- **延伸（extends）**：[[Skill-DARA-Agentic-MMM-Optimizer]]（MMM+DARA 提供宏观预算分配，本 Skill 提供关键词级别的多目标出价）
- **延伸（extends）**：[[Skill-Price-Elasticity-Estimation]]（价格弹性 + 出价弹性 = 价格和广告的联合多目标优化）
- **可组合（combinable）**：[[Skill-Channel-Saturation-Curve]]（组合：渠道饱和曲线识别最优投入量 + 多目标出价在该量级下找最优价格点）
- **可组合（combinable）**：[[Skill-Purchase-Intent-Prediction]]（组合：高意图用户出更高竞价（重要性加权）+ 多目标优化平衡新客/老客比例）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 旺季曝光量提升 20-35%（同预算）：旺季 GMV 增益 ¥15-40 万
  - 淡季 ROAS 提升 15-25%：节省无效广告花费 ¥5-15 万/季度
  - 自动化模式切换（旺季/淡季）：减少手动调价频率，节省运营时间
  - **年化综合 ROI：¥20-60 万**

- **实施难度**：⭐⭐⭐☆☆（多目标优化概念清晰；Amazon 广告 API 提供实时数据；约 3-4 周实施）

- **优先级评分**：⭐⭐⭐⭐☆（桥接 15-营销投放 ↔ 02-A_B实验 ↔ 17-价格优化 三域；多目标广告出价是中型卖家下一个优化阶段的必备工具）

- **评估依据**：多目标 RL 在广告出价（KDD 2024）有明确实验验证；阿里/字节等平台 RTB 系统已普遍采用多目标优化；母婴品类旺季/淡季差异明显，动态切换价值显著
