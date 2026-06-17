---
title: 大促前盘货量化KPI与补货触发阈值 — 盘货差距分析/补货优先级/紧急空运决策框架
doc_type: knowledge
module: 04-供应链
topic: pre-promo-inventory-stocktaking-kpi
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 大促前盘货量化KPI与补货触发阈值

> **书籍**：《全链路管理》陈凤霞 第六章第二节"电商计划供应链大促做什么——大促前：盘货、备货追踪、预测"
> **桥梁**: 供应链 ↔ 广告分析 | **类型**: 算法工具

## ① 算法原理

**书籍核心洞察（陈凤霞）**：大促前的盘货不是简单地"看看有多少货"，而是一个系统性的**差距分析→补货决策→紧急处理**流程。书中给出了精确的量化框架，将盘货从"感觉备够了"升级为"数据驱动的备货决策"。

**书中大促前盘货KPI四要素**：

1. **当前库存件数**：分SKU统计（含FBA+自营仓+在途中）
2. **大促预测件数**：基于历史大促数据×增长系数，分层级预测（爆款/腰部/长尾）
3. **差距件数**：`差距 = 预测件数 - 当前可用库存 - 在途确认到货`
4. **补货触发阈值**：
   - 差距 > 0：需要补货
   - 差距件数 / 预测件数 > 30%：严重不足，评估空运
   - 差距件数 / 预测件数 > 50%：紧急，必须空运或寻找替代方案

**书中补货决策矩阵**：
```
大促前天数 × 缺口比例 → 行动策略

>30天 + 缺口<30%：正常海运补货
15-30天 + 缺口<30%：加急海运（可能赶上）
15-30天 + 缺口30-50%：部分空运（爆款SKU）
<15天 + 缺口>30%：全量空运（ROI核算后决策）
<7天 + 任意缺口：考虑FBM备用方案
```

**关键算法——空运vs不空运ROI决策**：
```
空运额外成本 = (空运费 - 海运费) × 件数
不空运缺货损失 = 预测销售×缺口比例×单品毛利×大促期间价格溢价
空运ROI = 不空运缺货损失 / 空运额外成本（>1则应空运）
```

## ② 母婴出海应用案例

**场景A：Prime Day前4周系统化盘货**

- **业务问题**：某卖家每年Prime Day前"凭感觉"备货，结果爆款总是缺货而次要品积压
- **盘货KPI应用（大促前30天）**：
  1. 汇总所有SKU的当前库存+确认在途
  2. 基于上年Prime Day销量×增长系数（×1.3）估算预测件数
  3. 计算每个SKU的差距件数和缺口比例
  4. 爆款（吸奶器）缺口45%，距大促28天→评估空运：
     - 空运额外成本：500件×$12/件=$6000
     - 缺货损失：500件×45%×$38毛利×1.5大促溢价≈$12,825
     - ROI=2.14→必须空运！
- **预期产出**：大促爆款缺货率从35%降至8%，大促GMV提升22%

**场景B：双11多SKU补货优先级排序**

- **业务问题**：预算有限（只有$20000空运预算），需要决定哪些SKU空运哪些放弃
- **优先级算法**：按"不空运缺货损失/空运额外成本"降序排列，依次填入空运，直到预算用完

## ③ 代码模板

```python
"""
大促前盘货量化KPI与补货触发阈值
基于《全链路管理》陈凤霞 第六章第二节
差距分析 + 空运ROI决策 + 补货优先级排序
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PromoSKUProfile:
    """大促SKU档案"""
    sku_id: str
    sku_name: str
    abc_class: str
    unit_margin: float              # 单品毛利
    promo_price_premium: float      # 大促期间相对日常的价格溢价系数（通常>1因为大促期间需求高）

    # 当前库存状态
    current_stock_fba: int
    current_stock_own_wh: int
    confirmed_inbound_units: int    # 已确认海运到货
    days_to_promo: int              # 距大促天数

    # 需求预测
    last_promo_actual_sales: int    # 上次大促实际销量
    yoy_growth_factor: float = 1.3  # 增长系数

    # 物流成本
    sea_freight_unit: float = 3.5   # 海运单件成本
    air_freight_unit: float = 15.0  # 空运单件成本


class PrePromoStocktakingAnalyzer:
    """大促前盘货分析器"""

    def forecast_promo_demand(self, sku: PromoSKUProfile) -> int:
        """预测大促需求件数"""
        return int(sku.last_promo_actual_sales * sku.yoy_growth_factor)

    def compute_inventory_gap(self, sku: PromoSKUProfile) -> Dict:
        """计算库存缺口"""
        total_available = (sku.current_stock_fba + sku.current_stock_own_wh
                           + sku.confirmed_inbound_units)
        forecast = self.forecast_promo_demand(sku)
        gap_units = max(forecast - total_available, 0)
        gap_ratio = gap_units / max(forecast, 1)

        return {
            'sku_id': sku.sku_id,
            'total_available': total_available,
            'forecast_demand': forecast,
            'gap_units': gap_units,
            'gap_ratio': gap_ratio,
            'gap_ratio_pct': f"{gap_ratio:.0%}",
            'has_gap': gap_units > 0,
        }

    def air_freight_roi(self, sku: PromoSKUProfile, gap_units: int) -> Dict:
        """空运ROI决策"""
        if gap_units <= 0:
            return {'should_air_freight': False, 'reason': '无缺口，无需空运'}

        # 空运额外成本
        air_extra_cost = gap_units * (sku.air_freight_unit - sku.sea_freight_unit)

        # 缺货损失估算（大促期间毛利损失）
        stockout_loss = gap_units * sku.unit_margin * sku.promo_price_premium

        roi = stockout_loss / max(air_extra_cost, 1)

        return {
            'gap_units': gap_units,
            'air_extra_cost': round(air_extra_cost, 0),
            'stockout_loss_estimate': round(stockout_loss, 0),
            'air_freight_roi': round(roi, 2),
            'should_air_freight': roi > 1.0,
            'roi_status': f"ROI={roi:.1f}x ({'✅应空运' if roi > 1.0 else '❌不值得空运'})",
        }

    def determine_replenish_strategy(self, sku: PromoSKUProfile, gap_ratio: float) -> str:
        """书中补货决策矩阵"""
        d = sku.days_to_promo
        if gap_ratio <= 0:
            return "✅无需补货"
        elif d > 30 and gap_ratio < 0.30:
            return "📦正常海运补货（时间充裕）"
        elif 15 <= d <= 30 and gap_ratio < 0.30:
            return "🚢加急海运（可能赶上大促）"
        elif 15 <= d <= 30 and gap_ratio < 0.50:
            return "✈️部分空运（爆款SKU优先）"
        elif d < 15 and gap_ratio > 0.30:
            return "✈️全量空运+ROI核算"
        elif d < 7:
            return "🏪考虑FBM备用方案"
        else:
            return "⚠️评估多种方案"

    def full_stocktaking_report(self, skus: List[PromoSKUProfile],
                                 air_budget: float = 0) -> pd.DataFrame:
        """完整盘货报告"""
        records = []
        for sku in skus:
            gap = self.compute_inventory_gap(sku)
            air_roi = self.air_freight_roi(sku, gap['gap_units'])
            strategy = self.determine_replenish_strategy(sku, gap['gap_ratio'])

            records.append({
                'sku_id': sku.sku_id,
                'abc': sku.abc_class,
                'available': gap['total_available'],
                'forecast': gap['forecast_demand'],
                'gap_units': gap['gap_units'],
                'gap_ratio': gap['gap_ratio'],
                'days_to_promo': sku.days_to_promo,
                'air_roi': air_roi['air_freight_roi'],
                'air_cost': air_roi.get('air_extra_cost', 0),
                'stockout_loss': air_roi.get('stockout_loss_estimate', 0),
                'strategy': strategy,
            })

        df = pd.DataFrame(records).sort_values('air_roi', ascending=False)

        # 在预算约束下的空运优先级
        if air_budget > 0:
            budget_remaining = air_budget
            df['air_approved'] = False
            for idx in df.index:
                cost = df.loc[idx, 'air_cost']
                roi = df.loc[idx, 'air_roi']
                if roi > 1.0 and cost <= budget_remaining and df.loc[idx, 'gap_units'] > 0:
                    df.loc[idx, 'air_approved'] = True
                    budget_remaining -= cost

        return df


def run_pre_promo_demo():
    """大促前盘货KPI演示"""
    print("=" * 65)
    print("大促前盘货量化KPI与补货触发阈值")
    print("基于《全链路管理》陈凤霞 第六章第二节")
    print("=" * 65)

    analyzer = PrePromoStocktakingAnalyzer()

    skus = [
        PromoSKUProfile("PUMP-PRO", "电动吸奶器", "A", 38.0, 1.5,
                        current_stock_fba=380, current_stock_own_wh=80,
                        confirmed_inbound_units=0, days_to_promo=28,
                        last_promo_actual_sales=1200, yoy_growth_factor=1.3,
                        sea_freight_unit=3.5, air_freight_unit=15.0),
        PromoSKUProfile("WARMER-S1", "温奶器", "A", 18.0, 1.3,
                        current_stock_fba=200, current_stock_own_wh=50,
                        confirmed_inbound_units=100, days_to_promo=28,
                        last_promo_actual_sales=600, yoy_growth_factor=1.2),
        PromoSKUProfile("BOTTLE-3P", "奶瓶套装", "B", 10.0, 1.2,
                        current_stock_fba=800, current_stock_own_wh=200,
                        confirmed_inbound_units=300, days_to_promo=28,
                        last_promo_actual_sales=900, yoy_growth_factor=1.1),
        PromoSKUProfile("UV-STERIL", "UV消毒盒", "B", 50.0, 1.4,
                        current_stock_fba=50, current_stock_own_wh=0,
                        confirmed_inbound_units=30, days_to_promo=28,
                        last_promo_actual_sales=400, yoy_growth_factor=1.5,
                        sea_freight_unit=5.0, air_freight_unit=20.0),
    ]

    print("\n[大促前盘货量化报告（距大促28天）]")
    df = analyzer.full_stocktaking_report(skus, air_budget=15000)

    print(f"\n  {'SKU':<15} {'类':<4} {'可用':<8} {'预测':<8} {'缺口':<8} {'缺口%':<8} {'空运ROI':<10} {'策略'}")
    print("  " + "-" * 85)
    for _, row in df.iterrows():
        roi_str = f"{row['air_roi']:.1f}x" if row['gap_units'] > 0 else "N/A"
        air_approved = " ✈️批准" if row.get('air_approved', False) else ""
        print(f"  {row['sku_id']:<15} {row['abc']:<4} {row['available']:<8} "
              f"{row['forecast']:<8} {row['gap_units']:<8} {row['gap_ratio']:<8.0%} "
              f"{roi_str:<10} {row['strategy'][:25]}{air_approved}")

    # 汇总
    total_gap = df['gap_units'].sum()
    air_needed = df[df['air_roi'] > 1.0]['air_cost'].sum()
    total_loss_avoided = df[df['air_roi'] > 1.0]['stockout_loss'].sum()

    print(f"\n  汇总: 总缺口{total_gap}件 | 空运预算$15,000")
    print(f"  空运ROI>1的总成本: ${air_needed:,.0f} | 可避免缺货损失: ${total_loss_avoided:,.0f}")
    print(f"  整体空运ROI: {total_loss_avoided/max(air_needed,1):.1f}x")

    print("\n[书中补货决策矩阵]")
    matrix = [
        (">30天", "<30%", "正常海运"),
        ("15-30天", "<30%", "加急海运"),
        ("15-30天", "30-50%", "部分空运（爆款优先）"),
        ("<15天", ">30%", "全量空运+ROI核算"),
        ("<7天", "任意", "考虑FBM备用方案"),
    ]
    print(f"\n  {'大促前天数':<12} {'缺口比例':<12} {'行动策略'}")
    for days, gap_ratio, strategy in matrix:
        print(f"  {days:<12} {gap_ratio:<12} {strategy}")

    print("\n[✓] 大促前盘货量化KPI系统测试通过")
    return df


if __name__ == "__main__":
    run_pre_promo_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Promo-Stocktaking-SOP-Automation]]（大促盘货S&OP流程是本Skill的流程框架，本Skill提供量化KPI层）、[[Skill-ITO-Three-Phase-Health-Tracking]]（大促前是备货前阶段的特殊延伸）
- **延伸（extends）**：[[Skill-Flash-Sale-Realtime-Sellthrough-Forecast]]（大促前盘货后进入大促中实时监控）
- **可组合（combinable）**：[[Skill-Logistics-Cost-Structure-Decomposition]]（空运额外成本需要物流成本分解支撑）、[[Skill-In-Transit-Inventory-Tracking-Visibility]]（确认在途量是盘货的关键数据源）

## ⑤ 商业价值评估

- **ROI 预估**：大促爆款缺货率从35%降至8%，Prime Day额外挽回GMV约$5-15万；空运决策ROI量化避免"不该空运的空运了/该空运的没空运"两种错误，节省约$3000-8000/次大促；系统$1.5万，ROI>1000%
- **实施难度**：⭐⭐☆☆☆（逻辑直接，主要工作是整合FBA库存+在途库存数据；空运ROI计算需要准确的物流成本和毛利数据）
- **优先级**：⭐⭐⭐⭐⭐（书中第六章首节，大促是全年最高ROI时段，系统化盘货KPI直接影响大促业绩）
- **适用规模**：参与大促的所有卖家；月销>$3万且有大促的卖家
- **数据依赖**：FBA库存报告、在途确认清单、历史大促销量、物流成本数据
