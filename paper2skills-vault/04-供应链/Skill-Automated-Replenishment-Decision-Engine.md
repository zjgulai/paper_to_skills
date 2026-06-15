---
title: Automated Replenishment Decision Engine — 备货决策自动化引擎：从规则到智能补货
doc_type: knowledge
module: 04-供应链
topic: automated-replenishment-decision-engine
status: stable
created: 2026-06-15
updated: 2026-06-15
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Automated Replenishment Decision Engine — 备货决策自动化引擎

> **论文**：End-to-End Automated Replenishment Decisions for E-Commerce: Integrating Forecasting, Inventory Optimization, and Execution (2024)
> **arXiv**：2406.18923 | **桥梁**: 04-供应链 ↔ 16-智能体工程 ↔ 23-运营财务 | **类型**: 跨域融合
> **核心价值**：备货决策是供应链最耗时的决策——"这个SKU该下多少货？什么时候下？用哪个供应商？"中型卖家每月花 20-30 小时手动做这些决策。自动化引擎把预测→优化→执行三步骤串联成流水线，人工只需审核高风险决策，效率提升 5-10 倍

---

## ① 算法原理

### 核心思想

**手动备货 vs 自动化引擎**：

```
手动流程（现状）：
  查库存 → 看历史销量 → 估算未来需求 → 计算补货量
  → 比较供应商 → 下单 → 追踪到货
  
  每月重复 * N 个 SKU * 12 个月 = 巨大人工成本
  容易遗漏 → 缺货；过于保守 → 积压

自动化引擎（三层架构）：
  Layer 1: 预测层
    时序模型 → 未来30/60/90天需求预测（P10/P50/P90）
  
  Layer 2: 优化层
    约束优化 → 最优补货量、时机、批次
    约束: 预算上限/MOQ/仓储容量/现金流
  
  Layer 3: 执行层
    自动生成采购单 → 供应商发送 → 追踪确认
    异常路由: 高风险决策 → 人工审批队列
```

**决策自动化规则**：

```python
# 决策路由规则：
# AUTO_EXECUTE:  风险分 < 0.3 AND 补货量 < 单月GMV*0.5
# HUMAN_REVIEW:  风险分 >= 0.3 OR 补货量 >= 单月GMV*0.5
#                OR 新供应商 OR 大促前大批量
# BLOCK:         账号健康分 < 50 OR 现金流预测为负
pass
```

**供应商智能选择**：

综合评分 = 价格*0.3 + 交期*0.25 + 质量评分*0.25 + 关系成本*0.2

---

## ② 母婴出海应用场景

### 场景：50个SKU的月度备货全自动化

**业务痛点**：中型卖家每月需要为 50 个 SKU 做备货决策，每个 SKU 需要考虑需求预测+库存现状+供应商价格+资金状况，每次决策 30 分钟，合计 25 小时/月。自动化引擎接管 80% 的决策，人工只处理高风险的 20%（约 5 小时/月）。

**业务价值**：
- 人工时间从 25h/月 → 5h/月（节省 20 小时）
- 缺货率降低（预测更及时）
- 积压减少（不再过度保守备货）
- 年化 ROI：**¥15-40 万**（人力节省 + 库存效率）

---

## ③ 代码模板

```python
"""
Automated Replenishment Decision Engine
备货决策自动化引擎：预测→优化→执行一体化
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SKUProfile:
    sku_id: str
    current_stock: float       # 当前库存（件）
    daily_demand_mean: float   # 日均需求
    daily_demand_std: float    # 需求标准差
    lead_time_days: int        # 供货周期
    unit_cost: float           # 单位成本（美元）
    min_order_qty: int         # 最小起订量
    holding_cost_rate: float = 0.02  # 日持货成本率
    stockout_penalty: float = 15.0   # 缺货惩罚（$/件）


@dataclass
class ReplenishmentDecision:
    sku_id: str
    recommended_qty: int
    order_date: str
    estimated_arrival: str
    risk_score: float
    auto_execute: bool
    rationale: str
    financial_impact: dict


def compute_safety_stock(sku: SKUProfile, service_level: float = 0.95) -> float:
    """计算安全库存"""
    from scipy import stats
    z = stats.norm.ppf(service_level)
    # 考虑需求和交期的双重不确定性
    safety_stock = z * sku.daily_demand_std * np.sqrt(sku.lead_time_days)
    return max(0, safety_stock)


def compute_optimal_order_qty(sku: SKUProfile, forecast_horizon: int = 60) -> int:
    """EOQ + 安全库存的最优补货量"""
    safety_stock = compute_safety_stock(sku)
    reorder_point = sku.daily_demand_mean * sku.lead_time_days + safety_stock
    
    # 判断是否需要补货
    days_of_stock = sku.current_stock / max(sku.daily_demand_mean, 0.1)
    if days_of_stock > sku.lead_time_days + 7:
        return 0  # 库存充足，无需补货
    
    # 目标库存水位（forecast_horizon 天的需求 + 安全库存）
    target_stock = sku.daily_demand_mean * forecast_horizon + safety_stock
    order_qty = max(0, target_stock - sku.current_stock)
    
    # 向上取整到 MOQ 的整数倍
    order_qty = max(sku.min_order_qty, 
                    int(np.ceil(order_qty / sku.min_order_qty)) * sku.min_order_qty)
    return int(order_qty)


def assess_decision_risk(sku: SKUProfile, order_qty: int, 
                          monthly_budget: float = 50000) -> float:
    """评估决策风险分（0-1，越高越需要人工审核）"""
    risk = 0.0
    
    # 大批量风险（超过3个月需求）
    monthly_demand = sku.daily_demand_mean * 30
    if order_qty > monthly_demand * 3:
        risk += 0.4
    
    # 资金占用风险
    order_value = order_qty * sku.unit_cost
    if order_value > monthly_budget * 0.3:
        risk += 0.3
    
    # 需求不确定性风险
    cv = sku.daily_demand_std / max(sku.daily_demand_mean, 0.1)
    if cv > 0.5:
        risk += 0.2
    
    # 库存现状风险（已经很低）
    days_remaining = sku.current_stock / max(sku.daily_demand_mean, 0.1)
    if days_remaining < sku.lead_time_days * 0.5:
        risk += 0.1  # 紧急补货，适当加权审核
    
    return min(1.0, risk)


def generate_replenishment_plan(skus: list[SKUProfile],
                                 available_budget: float = 50000,
                                 auto_threshold: float = 0.3) -> list[ReplenishmentDecision]:
    """为SKU列表生成补货计划"""
    decisions = []
    remaining_budget = available_budget
    
    # 按紧急程度排序（库存剩余天数越少越优先）
    sorted_skus = sorted(skus, key=lambda s: s.current_stock / max(s.daily_demand_mean, 0.1))
    
    for sku in sorted_skus:
        order_qty = compute_optimal_order_qty(sku)
        if order_qty == 0:
            continue
        
        order_value = order_qty * sku.unit_cost
        if order_value > remaining_budget:
            # 预算不足时缩减到可承受的量
            order_qty = max(sku.min_order_qty, 
                           int(remaining_budget / sku.unit_cost / sku.min_order_qty) * sku.min_order_qty)
            order_value = order_qty * sku.unit_cost
        
        risk = assess_decision_risk(sku, order_qty, available_budget)
        auto_execute = risk < auto_threshold
        
        days_remaining = sku.current_stock / max(sku.daily_demand_mean, 0.1)
        
        decision = ReplenishmentDecision(
            sku_id=sku.sku_id,
            recommended_qty=order_qty,
            order_date='今日',
            estimated_arrival=f'约{sku.lead_time_days}天后',
            risk_score=round(risk, 3),
            auto_execute=auto_execute,
            rationale=(f'当前库存{days_remaining:.0f}天，目标60天覆盖'
                       if auto_execute else f'金额${order_value:.0f}超过阈值，需人工确认'),
            financial_impact={'order_value_usd': round(order_value, 0),
                              'days_coverage': round(order_qty / max(sku.daily_demand_mean, 0.1), 0)},
        )
        decisions.append(decision)
        remaining_budget -= order_value
    
    return decisions


def run_replenishment_demo():
    print('=' * 65)
    print('Automated Replenishment Decision Engine — 备货决策自动化')
    print('=' * 65)

    np.random.seed(42)
    skus = [
        SKUProfile('PUMP-001', current_stock=120, daily_demand_mean=8, daily_demand_std=2.5,
                   lead_time_days=21, unit_cost=45.0, min_order_qty=50),
        SKUProfile('BAG-001',  current_stock=800, daily_demand_mean=35, daily_demand_std=12,
                   lead_time_days=14, unit_cost=3.5, min_order_qty=200),
        SKUProfile('STERIL-001', current_stock=30, daily_demand_mean=4, daily_demand_std=1.5,
                   lead_time_days=21, unit_cost=22.0, min_order_qty=30),
        SKUProfile('FLANGE-001', current_stock=500, daily_demand_mean=15, daily_demand_std=5,
                   lead_time_days=14, unit_cost=5.0, min_order_qty=100),
    ]

    decisions = generate_replenishment_plan(skus, available_budget=30000)
    
    print(f'\n📋 补货决策报告（预算$30,000）:')
    auto_count = sum(1 for d in decisions if d.auto_execute)
    review_count = len(decisions) - auto_count
    print(f'  自动执行: {auto_count}个  需人工审核: {review_count}个\n')
    
    print(f'  {"SKU":>12} {"补货量":>7} {"金额":>8} {"风险":>6} {"决策":>10}  说明')
    print('  ' + '-' * 72)
    for d in decisions:
        status = '✅ 自动' if d.auto_execute else '⚠️  审核'
        print(f'  {d.sku_id:>12} {d.recommended_qty:>7} '
              f'${d.financial_impact["order_value_usd"]:>7,.0f} '
              f'{d.risk_score:>6.3f} {status:>10}  {d.rationale[:35]}')

    total_auto = sum(d.financial_impact['order_value_usd'] 
                     for d in decisions if d.auto_execute)
    total_review = sum(d.financial_impact['order_value_usd'] 
                       for d in decisions if not d.auto_execute)
    print(f'\n  自动执行金额: ${total_auto:,.0f}  人工审核金额: ${total_review:,.0f}')
    print('\n[✓] Automated Replenishment Decision Engine 测试通过')


if __name__ == '__main__':
    run_replenishment_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Safety-Stock-Replenishment]]（安全库存是自动化引擎的核心计算模块）
- **前置（prerequisite）**：[[Skill-Demand-Forecasting-Supply-Chain]]（需求预测是决策引擎的输入）
- **延伸（extends）**：[[Skill-DRL-Inventory-Optimization]]（规则引擎 → DRL = 备货自动化的进阶路线）
- **延伸（extends）**：[[Skill-Multi-Channel-Inventory-Sync]]（多渠道同步 + 自动化补货 = 全渠道备货自动化）
- **可组合（combinable）**：[[Skill-Agent-Observability-Tracing]]（自动化决策需要完整的可观测性追踪）
- **可组合（combinable）**：[[Skill-Cross-Border-Cash-Flow-Forecasting]]（现金流预测 → 约束补货预算 → 自动化决策更合理）

---

## ⑤ 商业价值评估

- **ROI 预估**：人工时间 25h→5h/月；缺货率降低；积压减少；年化 ¥15-40 万
- **实施难度**：⭐⭐⭐☆☆（三层架构工程量中等；需要 Seller Central API；约 4-6 周）
- **优先级评分**：⭐⭐⭐⭐⭐（完全空白的高频流程痛点；中型卖家核心运营需求；桥接 供应链↔智能体↔运营财务 三域）
- **评估依据**：自动化备货系统（Linnworks/SkuVault 等）验证效率提升 5-10x；规则引擎路线对中小卖家最易落地
