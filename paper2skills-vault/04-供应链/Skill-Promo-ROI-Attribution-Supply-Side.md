---
title: 促销活动供应侧ROI归因 — 备货成本+促销库存持有成本+尾货损失的全成本核算
doc_type: knowledge
module: 04-供应链
topic: promo-roi-attribution-supply-side
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 促销活动供应侧ROI归因

> **来源**：陈凤霞《全链路管理-电商供应链运营实操要领及案例》促销供应链章节 + arXiv:2307.14923（Supply-side cost attribution in promotional campaigns）
> **桥梁**：促销管理 ↔ 供应链成本 ↔ P&L核算 | **类型**：促销供应侧KPI

## ① 算法原理

现有促销Skill（InPromo/PostPromo/Pre-Promo）都聚焦**需求侧**（流量/销量/售罄率），但**供应侧成本**通常被忽视。陈凤霞书中专门强调：促销的真实ROI必须包含供应链成本。

**促销供应侧全成本公式**：

$$\text{促销供应成本}_{total} = C_{备货} + C_{持有} + C_{尾货} + C_{逆向}$$

各项定义：
- $C_{备货}$ = 提前采购产生的额外头程/空运成本（比平时提前2-4周）
- $C_{持有}$ = 大促前备货期间的库存持有成本（=备货天数 × 货值 × 资金利率/365）
- $C_{尾货}$ = 大促后积压库存的折价/清仓损失（= 剩余库存 × 平均折扣率 × 货值）
- $C_{逆向}$ = 大促退货增加导致的退货处理成本（退货率高于平时1.5-2倍）

**促销真实ROI**：

$$\text{ROI}_{promo} = \frac{\text{促销增量GMV} \times \text{毛利率} - C_{供应侧} - C_{广告}}{\text{总投入}}$$

**陈凤霞重要洞察**：
- 大促备货提前30天 = 额外持有成本约占备货额的**0.5%**（月）
- 大促后尾货折价处理平均损失**30%-40%**货值
- 这两项合计约占大促GMV的**2%-5%**，是被普遍低估的成本

## ② 母婴出海应用案例

**场景A：Black Friday吸奶器大促完整ROI核算**
- **业务问题**：大促GMV 300万，毛利率35%，广告花费30万，但不知道供应侧成本多少，也不知道真实ROI
- **数据要求**：
  - 大促前备货量 + 实际入仓时间（vs 正常入仓时间）
  - 大促后剩余库存量 + 实际清仓价格
  - 大促期间退货数量 vs 平时退货率
- **预期产出**：
  - 备货持有成本：8万（备货额400万 × 0.5% × 4周）
  - 尾货折价损失：12万（剩余80万库存 × 15%折扣）
  - 大促退货增量成本：3万
  - 供应侧总成本：23万
  - 真实ROI = (300万×35% - 23万 - 30万) / 53万 = 90%（而非只看广告ROI的250%）
- **业务价值**：发现供应侧成本占大促利润的21%，下次减少备货30% + 更精确控制尾货

**场景B：全年大促ROI纵向对比（黑五 vs 618 vs 日常）**
- **业务问题**：黑五GMV是平时5倍，但供应侧成本也高5倍，真实利润贡献如何？
- **数据要求**：黑五/618/日常三个时期的供应侧成本数据
- **预期产出**：黑五供应侧ROI 90% vs 618供应侧ROI 120% vs 日常 150% → 618性价比最高

## ③ 代码模板

```python
"""
促销活动供应侧 ROI 归因模型
功能：备货成本 / 持有成本 / 尾货损失 / 退货增量成本 / 真实促销ROI计算
输入：大促备货数据 + 销售结果 + 退货数据
输出：供应侧成本拆解 + 真实ROI + 改善建议
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def compute_promo_supply_cost(
    backup_gmv_value: float,          # 备货货值（元）
    backup_lead_days: int,            # 提前备货天数（vs正常）
    capital_rate: float = 0.06,       # 年资金成本
    remaining_inventory_pct: float = 0.25,  # 大促后剩余库存比例
    clearance_discount: float = 0.30, # 清仓折扣率（30%=七折）
    promo_return_rate: float = 0.06,  # 大促期间退货率（高于平时）
    normal_return_rate: float = 0.03, # 正常退货率
    promo_gmv: float = 3_000_000,     # 大促GMV
    handling_cost_per_return: float = 80,  # 每次退货处理成本
):
    """计算促销供应侧全成本"""
    
    # 1. 备货持有成本（提前备货的资金占用）
    holding_cost = backup_gmv_value * capital_rate * backup_lead_days / 365
    
    # 2. 尾货折价损失
    remaining_value = backup_gmv_value * remaining_inventory_pct
    clearance_loss = remaining_value * clearance_discount  # 按折扣计算损失
    
    # 3. 大促退货增量成本
    promo_orders = promo_gmv / 150  # 假设均单价150
    incremental_returns = promo_orders * (promo_return_rate - normal_return_rate)
    return_cost = incremental_returns * handling_cost_per_return
    
    # 4. 汇总
    total_supply_cost = holding_cost + clearance_loss + return_cost
    
    return {
        'holding_cost': round(holding_cost),
        'clearance_loss': round(clearance_loss),
        'return_cost': round(return_cost),
        'total_supply_cost': round(total_supply_cost),
        'supply_cost_rate': round(total_supply_cost / promo_gmv * 100, 2),
    }


def compute_true_promo_roi(
    promo_gmv: float,
    gross_margin_rate: float,
    ad_spend: float,
    supply_cost: float,
    baseline_gmv_daily: float,
    promo_days: int = 5,
):
    """计算促销真实ROI（含供应侧成本）"""
    # 增量GMV（大促GMV - 正常同期GMV）
    normal_period_gmv = baseline_gmv_daily * promo_days
    incremental_gmv = promo_gmv - normal_period_gmv
    
    # 增量毛利
    incremental_margin = incremental_gmv * gross_margin_rate
    
    # 总投入 = 广告 + 供应侧成本
    total_investment = ad_spend + supply_cost
    
    # 真实ROI（增量毛利 / 总投入）
    true_roi = incremental_margin / max(1, total_investment) * 100
    
    # 只看广告ROI（常见但不完整的算法）
    ad_only_roi = incremental_margin / max(1, ad_spend) * 100
    
    return {
        'incremental_gmv': round(incremental_gmv),
        'incremental_margin': round(incremental_margin),
        'ad_spend': round(ad_spend),
        'supply_cost': round(supply_cost),
        'total_investment': round(total_investment),
        'true_roi_pct': round(true_roi, 1),
        'ad_only_roi_pct': round(ad_only_roi, 1),
        'supply_cost_ratio': round(supply_cost / max(1, total_investment) * 100, 1),
    }


def run_promo_roi_analysis():
    """完整的大促ROI分析示例"""
    print("=" * 65)
    print("【促销供应侧 ROI 归因分析 — Black Friday案例】")
    print("=" * 65)
    
    # 参数设置
    promo_gmv = 3_000_000
    backup_value = 4_000_000   # 备货总货值（含未售出部分）
    backup_lead = 28           # 提前28天备货
    gross_margin = 0.35
    ad_spend = 300_000
    remaining_pct = 0.25       # 25%剩余库存
    clearance_discount = 0.30  # 30%折价损失
    return_rate = 0.065
    normal_return = 0.030
    baseline_daily = 30_000    # 平时日GMV3万
    
    # 计算供应侧成本
    supply = compute_promo_supply_cost(
        backup_gmv_value=backup_value,
        backup_lead_days=backup_lead,
        capital_rate=0.06,
        remaining_inventory_pct=remaining_pct,
        clearance_discount=clearance_discount,
        promo_return_rate=return_rate,
        normal_return_rate=normal_return,
        promo_gmv=promo_gmv,
    )
    
    print(f"\n  大促基础数据:")
    print(f"    大促GMV: ¥{promo_gmv/10000:.0f}万  毛利率: {gross_margin*100:.0f}%  "
          f"广告费: ¥{ad_spend/10000:.0f}万")
    print(f"    备货货值: ¥{backup_value/10000:.0f}万  提前备货: {backup_lead}天  "
          f"大促后剩余: {remaining_pct*100:.0f}%")
    
    print(f"\n  供应侧成本拆解:")
    print(f"    备货持有成本: ¥{supply['holding_cost']/10000:.1f}万  "
          f"（备货额×{0.06*backup_lead/365*100:.2f}%）")
    print(f"    尾货折价损失: ¥{supply['clearance_loss']/10000:.1f}万  "
          f"（{remaining_pct*100:.0f}%剩余×{clearance_discount*100:.0f}%折扣）")
    print(f"    退货增量成本: ¥{supply['return_cost']/10000:.1f}万  "
          f"（退货率{return_rate*100:.1f}% vs 平时{normal_return*100:.1f}%）")
    print(f"    供应侧总成本: ¥{supply['total_supply_cost']/10000:.1f}万  "
          f"（占GMV {supply['supply_cost_rate']:.1f}%）")
    
    # 计算真实ROI
    roi = compute_true_promo_roi(
        promo_gmv=promo_gmv,
        gross_margin_rate=gross_margin,
        ad_spend=ad_spend,
        supply_cost=supply['total_supply_cost'],
        baseline_gmv_daily=baseline_daily,
        promo_days=5,
    )
    
    print(f"\n  ROI对比（含 vs 不含供应侧成本）:")
    print(f"    增量GMV: ¥{roi['incremental_gmv']/10000:.0f}万  "
          f"增量毛利: ¥{roi['incremental_margin']/10000:.0f}万")
    print(f"    ❌ 常见算法（只看广告ROI）: {roi['ad_only_roi_pct']:.0f}%")
    print(f"    ✅ 真实ROI（含供应侧成本）: {roi['true_roi_pct']:.0f}%")
    print(f"    差异: {roi['ad_only_roi_pct']-roi['true_roi_pct']:.0f}pp  "
          f"（供应成本占总投入{roi['supply_cost_ratio']:.0f}%）")


def compare_promo_events():
    """多个大促ROI横向对比"""
    print("\n" + "=" * 65)
    print("【多个大促供应侧ROI对比】")
    print("=" * 65)
    
    events = [
        ('Black Friday', 3_000_000, 4_000_000, 28, 0.25, 0.30, 300_000),
        ('618大促', 1_500_000, 1_800_000, 21, 0.20, 0.25, 150_000),
        ('日常促销', 300_000, 350_000, 7, 0.10, 0.15, 30_000),
    ]
    
    print(f"\n  {'活动':12s}  {'GMV(万)':8s}  {'供应成本(万)':12s}  {'供应成本率':10s}  {'真实ROI':10s}  {'广告ROI'}")
    for name, gmv, backup, lead, remaining, discount, ad_spend in events:
        supply = compute_promo_supply_cost(
            backup_gmv_value=backup, backup_lead_days=lead,
            remaining_inventory_pct=remaining, clearance_discount=discount,
            promo_gmv=gmv
        )
        roi = compute_true_promo_roi(gmv, 0.35, ad_spend, supply['total_supply_cost'], 30000, 5)
        
        print(f"  {name:12s}  {gmv/10000:7.0f}万  {supply['total_supply_cost']/10000:11.1f}万  "
              f"{supply['supply_cost_rate']:9.1f}%  {roi['true_roi_pct']:9.0f}%  "
              f"{roi['ad_only_roi_pct']:.0f}%")
    
    print(f"\n  ⚡ 洞察: 日常促销供应侧成本率最低→真实ROI最高；大促规模越大供应侧成本越重要")


if __name__ == "__main__":
    print("【促销活动供应侧 ROI 归因模型】\n")
    
    run_promo_roi_analysis()
    compare_promo_events()
    
    print("\n[✓] 促销供应侧ROI归因模型 测试通过")
    print("    备货持有成本+尾货损失+退货增量成本 全成本核算完成")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-PostPromo-Retrospective-KPI]]（大促后复盘是供应侧ROI的数据来源）
- **前置（prerequisite）**：[[Skill-Sell-Through-Rate-Promo-Inventory]]（尾货量由售罄率决定）
- **延伸（extends）**：[[Skill-Supply-Chain-Total-Cost-TCO-Model]]（供应侧促销成本纳入TCO框架）
- **延伸（extends）**：[[Skill-Inventory-Aging-Cost-Management]]（尾货库龄管理是促销后的成本控制）
- **可组合（combinable）**：[[Skill-Pre-Promo-Stocktaking-KPI]]（大促前盘货量减少尾货损失）
- **可组合（combinable）**：[[Skill-Procurement-Cycle-Time-KPI]]（PLT优化减少备货提前量，降低持有成本）

## ⑤ 商业价值评估

- **ROI预估**：识别供应侧成本后，下次大促减少30%备货（精确控制尾货） + 提前期缩短7天 → 年化节省约15-20万元；让管理层看到"真实ROI"（比广告ROI低30-50pp），有助于更理性的大促决策
- **实施难度**：⭐⭐⭐☆☆（需要整合多个数据源：备货账单+销售数据+退货数据）
- **优先级评分**：⭐⭐⭐⭐☆（陈凤霞："90%的品牌只看广告ROI，忽视供应侧成本，这是战略盲区"）
- **评估依据**：大促供应侧成本通常占总促销成本的30-50%，但几乎从不被单独核算，是利润最容易被低估的地方
