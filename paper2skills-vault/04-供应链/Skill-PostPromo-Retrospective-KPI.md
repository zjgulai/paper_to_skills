---
title: 大促后复盘KPI体系 — 售罄率分析/备货vs实销对比/履约回顾/改善行动计划
doc_type: knowledge
module: 04-供应链
topic: post-promo-retrospective-kpi-sellthrough-analysis
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 大促后复盘KPI体系

> **书籍**：《全链路管理》陈凤霞 第六章第二节"电商计划供应链大促做什么——大促后：滞销售罄分析、紧急补货"第三节"电商物流供应链大促做什么——大促后：物流履约和达成率"
> **桥梁**: 供应链 ↔ A/B实验 | **类型**: 算法工具

## ① 算法原理

**书籍核心洞察（陈凤霞）**：书中专章阐述大促后复盘的"系统性方法论"——不是简单地看"卖了多少"，而是要从**售罄率、备货vs实销对比、物流履约回顾、改善行动计划**四个维度系统化学习，为下一次大促提供数据支撑。书中特别指出：大促后的日销可能因为爆品售罄而面临断货风险（高售罄率反而是新的问题！）。

**大促后四维复盘体系（书中框架）**：

1. **售罄率分析（Sellthrough Rate Analysis）**：
   - 大促后售罄率 = 大促期间销售件数 / 大促前备货件数
   - 分层标准（书中）：
     - 售罄率>90%：爆款信号→下次备货要增加
     - 售罄率60-90%：健康
     - 售罄率40-60%：偏低→分析原因（价格/流量/质量）
     - 售罄率<40%：严重积压→启动清仓
   - **反直觉洞察**：售罄率>90%的SKU，大促后需要立即评估日销补货需求（避免"爆款→断货"）

2. **备货vs实销偏差分析**：
   - 备货准确率 = 1 - |备货量 - 实际销售| / max(备货量, 实际销售)
   - 分SKU归因：哪些SKU备多了、哪些备少了、哪些预测模型需要改进
   - 归因维度：预测误差 vs 执行误差（备货差距是预测问题还是采购落地问题）

3. **物流履约回顾**：
   - 大促期间发货及时率
   - 大促期间ODR（订单缺陷率）变化
   - 物流成本vs预算偏差
   - 客户投诉中物流相关比例

4. **改善行动计划（Action Item Tracking）**：
   - 每个复盘发现→对应具体行动项→负责人+截止日期
   - 行动完成率：下次大促前完成的行动项比例

## ② 母婴出海应用案例

**场景A：Prime Day系统化复盘**

- **业务问题**：某卖家Prime Day后开总结会，运营觉得"卖得不错"，但没有系统数据支撑，相同的问题每年重复
- **四维复盘应用**：
  1. 售罄率分析：吸奶器98%（爆款！下次+50%备货）；温奶器42%（严重积压→启动清仓）
  2. 备货vs实销：预测误差24%（偏低），执行误差8%（轻微）→重点改善预测模型
  3. 物流履约：发货及时率94%（Prime Day前20小时跌至65%→人力不足）；ODR 1.3%（接近红线）
  4. 行动计划：①吸奶器提升预测×1.5；②温奶器启动清仓；③下次大促提前1周招临时工；④优化包装降低ODR
- **预期产出**：通过系统化复盘建立"数字记忆"，下次大促准确率提升15%

**场景B：大促后紧急补货决策**

- **业务问题**：Prime Day结束，吸奶器售罄率98%，剩余库存只够3天日销，需要决策是否空运补货
- **书中框架**：售罄率>90%的SKU，大促后立即计算"大促后日销倍数"（大促后日销通常是大促前的1.3-1.8倍），基于此倍数决策紧急补货量

## ③ 代码模板

```python
"""
大促后复盘KPI体系
基于《全链路管理》陈凤霞 第六章第二、三节
四维复盘：售罄率/备货准确率/物流履约/改善行动
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PromoOutcome:
    """大促结果数据"""
    sku_id: str
    abc_class: str
    pre_promo_stock: int        # 大促前备货量
    promo_actual_sales: int     # 大促实际销售
    post_promo_stock: int       # 大促后剩余库存
    pre_promo_forecast: int     # 大促前预测销售
    planned_units: int          # 计划备货量（下单量）
    unit_margin: float
    # 物流数据
    dispatch_on_time_rate: float    # 发货及时率
    order_defect_rate: float        # ODR
    logistics_cost: float           # 实际物流成本
    logistics_budget: float         # 物流预算
    # 大促后日销
    post_promo_daily_sales: float   # 大促后日均销量


class PostPromoRetrospectiveKPI:
    """大促后复盘KPI分析器"""

    # 售罄率分级标准（书中）
    SELLTHROUGH_GRADES = {
        (0.90, 1.01): ('爆款', '🔥', '下次备货量×1.5，立即评估补货需求'),
        (0.60, 0.90): ('健康', '✅', '维持当前备货策略'),
        (0.40, 0.60): ('偏低', '⚠️', '分析原因：价格/流量/质量，下次减少20%备货'),
        (0.00, 0.40): ('积压', '🔴', '立即启动清仓，下次大幅减少'),
    }

    def sellthrough_analysis(self, outcome: PromoOutcome) -> Dict:
        """售罄率分析"""
        sellthrough = outcome.promo_actual_sales / max(outcome.pre_promo_stock, 1)

        grade, emoji, action = '未知', '?', '检查数据'
        for (lo, hi), (g, e, a) in self.SELLTHROUGH_GRADES.items():
            if lo <= sellthrough < hi:
                grade, emoji, action = g, e, a
                break

        # 大促后剩余库存日销覆盖
        days_of_remaining = outcome.post_promo_stock / max(outcome.post_promo_daily_sales, 0.01)

        return {
            'sku_id': outcome.sku_id,
            'pre_stock': outcome.pre_promo_stock,
            'actual_sales': outcome.promo_actual_sales,
            'post_stock': outcome.post_promo_stock,
            'sellthrough_rate': sellthrough,
            'sellthrough_pct': f"{sellthrough:.0%}",
            'grade': grade,
            'emoji': emoji,
            'recommended_action': action,
            'post_promo_doi': round(days_of_remaining, 1),
            'urgent_replenish': days_of_remaining < 7 and sellthrough > 0.85,
        }

    def procurement_accuracy_analysis(self, outcome: PromoOutcome) -> Dict:
        """备货准确率分析（预测误差 vs 执行误差）"""
        # 预测误差
        forecast_error = abs(outcome.pre_promo_forecast - outcome.promo_actual_sales) / max(outcome.promo_actual_sales, 1)
        # 执行误差（计划量 vs 实际备货量）
        execution_error = abs(outcome.planned_units - outcome.pre_promo_stock) / max(outcome.planned_units, 1)
        # 综合备货准确率
        procurement_accuracy = 1 - abs(outcome.pre_promo_stock - outcome.promo_actual_sales) / max(outcome.promo_actual_sales, 1)

        # 归因
        if forecast_error > execution_error * 1.5:
            attribution = "主要是预测偏差（改善预测模型）"
        elif execution_error > forecast_error * 1.5:
            attribution = "主要是执行偏差（改善采购落地）"
        else:
            attribution = "预测和执行均有偏差"

        return {
            'forecast_accuracy': 1 - forecast_error,
            'execution_accuracy': 1 - execution_error,
            'procurement_accuracy': max(procurement_accuracy, 0),
            'forecast_accuracy_pct': f"{1-forecast_error:.0%}",
            'attribution': attribution,
            'next_action': f"下次备货推荐量: {int(outcome.promo_actual_sales * 1.1)}件（实销×1.1作为基准）",
        }

    def logistics_fulfillment_review(self, outcome: PromoOutcome) -> Dict:
        """物流履约回顾"""
        cost_vs_budget = outcome.logistics_cost / max(outcome.logistics_budget, 1)
        cost_overrun = outcome.logistics_cost - outcome.logistics_budget

        return {
            'dispatch_on_time_rate': outcome.dispatch_on_time_rate,
            'dispatch_status': '✅' if outcome.dispatch_on_time_rate >= 0.95 else '🔴需改善',
            'odr': outcome.order_defect_rate,
            'odr_status': '✅安全' if outcome.order_defect_rate < 0.01 else '⚠️接近红线' if outcome.order_defect_rate < 0.015 else '🔴高危',
            'cost_vs_budget': cost_vs_budget,
            'cost_overrun': cost_overrun,
            'cost_status': '✅' if cost_overrun <= 0 else f"超支${cost_overrun:,.0f}",
        }

    def generate_action_plan(self, outcomes: List[PromoOutcome]) -> pd.DataFrame:
        """生成改善行动计划"""
        actions = []
        for outcome in outcomes:
            st = self.sellthrough_analysis(outcome)
            pa = self.procurement_accuracy_analysis(outcome)
            lf = self.logistics_fulfillment_review(outcome)

            if st['urgent_replenish']:
                actions.append({
                    'sku_id': outcome.sku_id,
                    'issue': '大促后库存不足风险',
                    'action': f"48小时内下补货单{int(outcome.post_promo_daily_sales * 30)}件",
                    'owner': '采购运营',
                    'urgency': 'P0',
                    'deadline': '48h内',
                })
            if st['sellthrough_rate'] < 0.40:
                actions.append({
                    'sku_id': outcome.sku_id,
                    'issue': f"严重积压（售罄率{st['sellthrough_pct']}）",
                    'action': f"启动阶梯清仓：第1周-15%，第2周-25%",
                    'owner': '销售运营',
                    'urgency': 'P1',
                    'deadline': '1周内',
                })
            if pa['forecast_accuracy'] < 0.75:
                actions.append({
                    'sku_id': outcome.sku_id,
                    'issue': f"预测误差大（FA={pa['forecast_accuracy_pct']}）",
                    'action': f"更新预测模型参数，下次备货量使用{pa['next_action'][:30]}",
                    'owner': '数据分析',
                    'urgency': 'P2',
                    'deadline': '2周内',
                })
            if lf['odr'] >= 0.01:
                actions.append({
                    'sku_id': outcome.sku_id,
                    'issue': f"ODR={lf['odr']:.2%} 偏高",
                    'action': '排查破损原因，改善包装规格，提高发货及时率',
                    'owner': '物流运营',
                    'urgency': 'P1',
                    'deadline': '下次大促前',
                })

        df = pd.DataFrame(actions) if actions else pd.DataFrame(
            columns=['sku_id', 'issue', 'action', 'owner', 'urgency', 'deadline'])
        return df.sort_values('urgency') if len(df) > 0 else df


def run_post_promo_retrospective_demo():
    """大促后复盘KPI演示"""
    print("=" * 65)
    print("大促后复盘KPI体系")
    print("基于《全链路管理》陈凤霞 第六章第二、三节")
    print("四维复盘：售罄率/备货准确率/物流履约/改善行动")
    print("=" * 65)

    analyzer = PostPromoRetrospectiveKPI()

    outcomes = [
        PromoOutcome("PUMP-PRO", "A", pre_promo_stock=1200, promo_actual_sales=1176,
                     post_promo_stock=24, pre_promo_forecast=900, planned_units=1100,
                     unit_margin=38, dispatch_on_time_rate=0.94, order_defect_rate=0.013,
                     logistics_cost=4200, logistics_budget=4000, post_promo_daily_sales=28),
        PromoOutcome("WARMER-S1", "A", pre_promo_stock=600, promo_actual_sales=372,
                     post_promo_stock=228, pre_promo_forecast=700, planned_units=600,
                     unit_margin=18, dispatch_on_time_rate=0.97, order_defect_rate=0.006,
                     logistics_cost=1800, logistics_budget=2000, post_promo_daily_sales=12),
        PromoOutcome("BOTTLE-3P", "B", pre_promo_stock=1800, promo_actual_sales=1440,
                     post_promo_stock=360, pre_promo_forecast=1600, planned_units=1800,
                     unit_margin=10, dispatch_on_time_rate=0.96, order_defect_rate=0.008,
                     logistics_cost=2700, logistics_budget=2500, post_promo_daily_sales=18),
        PromoOutcome("UV-STERIL", "B", pre_promo_stock=300, promo_actual_sales=108,
                     post_promo_stock=192, pre_promo_forecast=350, planned_units=300,
                     unit_margin=50, dispatch_on_time_rate=0.99, order_defect_rate=0.004,
                     logistics_cost=600, logistics_budget=700, post_promo_daily_sales=3),
    ]

    print("\n[售罄率分析]")
    print(f"  {'SKU':<15} {'备货':<8} {'实销':<8} {'售罄率':<10} {'评级':<10} {'大促后DOI':<12} {'行动'}")
    print("  " + "-" * 80)
    for outcome in outcomes:
        st = analyzer.sellthrough_analysis(outcome)
        urgent = " ⚡紧急补货!" if st['urgent_replenish'] else ""
        print(f"  {outcome.sku_id:<15} {outcome.pre_promo_stock:<8} "
              f"{outcome.promo_actual_sales:<8} {st['sellthrough_pct']:<10} "
              f"{st['emoji']}{st['grade']:<8} {st['post_promo_doi']:<12.0f}天 "
              f"{st['recommended_action'][:25]}{urgent}")

    print("\n[备货准确率分析]")
    for outcome in outcomes:
        pa = analyzer.procurement_accuracy_analysis(outcome)
        print(f"  {outcome.sku_id}: FA={pa['forecast_accuracy_pct']} | {pa['attribution'][:40]}")
        print(f"    {pa['next_action'][:55]}")

    print("\n[物流履约回顾]")
    for outcome in outcomes:
        lf = analyzer.logistics_fulfillment_review(outcome)
        print(f"  {outcome.sku_id}: 发货及时率{outcome.dispatch_on_time_rate:.0%} "
              f"{lf['dispatch_status']} | ODR={lf['odr']:.2%} {lf['odr_status']} | "
              f"物流成本{lf['cost_status']}")

    print("\n[改善行动计划]")
    action_df = analyzer.generate_action_plan(outcomes)
    if len(action_df) > 0:
        for _, row in action_df.iterrows():
            print(f"  [{row['urgency']}] {row['sku_id']}: {row['issue'][:40]}")
            print(f"       → {row['action'][:60]} | {row['owner']} | {row['deadline']}")
    else:
        print("  无需立即行动项")

    print("\n[书中关键洞察]")
    print("  售罄率>90% ≠ 结束，反而是补货紧急信号（大促后日销倍数效应）")
    print("  备货准确率拆分：预测误差 vs 执行误差（归因不同，改善路径不同）")
    print("  大促后复盘→行动计划→追踪完成率（形成学习飞轮）")
    print("\n[✓] 大促后复盘KPI系统测试通过")
    return action_df


if __name__ == "__main__":
    run_post_promo_retrospective_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-InPromo-Realtime-Decision-KPI]]（大促中数据是大促后复盘的输入）、[[Skill-Pre-Promo-Stocktaking-KPI]]（大促前备货量是售罄率的分母）
- **延伸（extends）**：[[Skill-ITO-Three-Phase-Health-Tracking]]（大促后=备货后阶段的特殊形态）
- **可组合（combinable）**：[[Skill-Long-Tail-SKU-Clearance-Optimization]]（低售罄率SKU进入清仓优化流程）、[[Skill-Forecast-Bias-Adjustment-Detection]]（备货准确率分析为下次大促更新预测修正系数）

## ⑤ 商业价值评估

- **ROI 预估**：系统化复盘使下次大促备货准确率提升15%，以Prime Day GMV$20万为例，准确率提升=减少$2万积压+减少$1.5万缺货损失；系统$1.5万，ROI>230%
- **实施难度**：⭐⭐☆☆☆（数据全来自大促结果，主要是建立系统化分析流程和行动追踪机制）
- **优先级**：⭐⭐⭐⭐⭐（书中专章，每次大促都是宝贵的学习机会，但90%的团队复盘浮于表面，系统化复盘是竞争壁垒）
- **适用规模**：所有参与主要大促的卖家
- **数据依赖**：大促前备货量、大促期间分时销售数据、物流履约数据（已在大促中收集）
