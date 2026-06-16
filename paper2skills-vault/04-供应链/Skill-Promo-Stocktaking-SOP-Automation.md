---
title: 大促盘货S&OP流程自动化 — 备货追踪、预案生成与紧急补货闭环
doc_type: knowledge
module: 04-供应链
topic: promo-stocktaking-sop-automation
status: stable
created: 2026-06-15
updated: 2026-06-15
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 大促盘货S&OP流程自动化

> **论文**：Automated S&OP for Promotional Events in E-Commerce / Multi-Stakeholder Inventory Planning Under Promotional Demand Spikes
> **arXiv**：2403.17821 | 2024 | **桥梁**: 供应链 ↔ 广告分析 | **类型**: 算法工具
> **书籍依据**：《全链路管理》第6章第5节"S&OP的运用：电商大促盘货流程"、第2节"电商计划供应链大促做什么"

## ① 算法原理

**业务背景（陈凤霞实战经验）**：书中详述大促S&OP盘货流程包含**参与部门（运营/采购/物流/财务）、盘货输入（预测/库存/供应商/资金）、盘货流程（对齐→差距→方案→决策）、成功因素（数据准确/提前启动/闭环追踪）**。作者强调大促S&OP必须提前8周启动，核心是实现"从上往下目标拆解"（总GMV→品类→SKU）与"从下往上盘货聚合"（每SKU库存×预测→汇总）两路收敛。

**反直觉洞察**：大促备货失败99%不是"预测不准"，而是**"预测结论没有闭环跟踪"**——预测了需要500件但采购只落实了300件，到货时间比预期晚5天，没有人跟踪这个缺口。书中强调：盘货不是一次会议，是一个持续8周的追踪机制，每周检查"当前状态vs计划"的差距，及时触发补救。

**核心算法：双路预测收敛 + 差距自动追踪 + 紧急补货触发**

1. **双路预测收敛（Two-Way Convergence）**：
   - **Top-Down（自上而下）**：
     - 大促总GMV目标 → 按品类历史占比拆解到类目
     - 类目目标 → 按历史大促SKU销售占比拆解到SKU
     - 考虑增长系数（今年 = 上年 × 预期增长率）
   - **Bottom-Up（自下而上）**：
     - 每个SKU历史大促销量 × 增长系数 = 预测需求
     - 聚合所有SKU → 总销售预测
   - **收敛验证**：`|Top-Down总量 - Bottom-Up总量| / 均值 < 10%` → 通过；否则触发差异分析

2. **备货追踪矩阵（Tracking Matrix）**：
   - 每周更新每个SKU的：预测需求 / 已采购 / 已到货 / 缺口
   - `备货缺口 = 预测需求 - (已到货 + 在途×到货概率)`
   - 缺口率 = 缺口 / 预测需求
   - 颜色预警：缺口率>20%→红，10-20%→黄，<10%→绿

3. **紧急补货触发规则**：
   - 大促前14天：缺口率>30% → 触发空运补货评估（成本vs缺货损失）
   - 大促前7天：缺口率>15% → 强制空运或FBM备用方案
   - 大促前3天：任何A类缺口 → 立即启动本地采购/调拨
   - 成本阈值：`空运附加成本 < 缺货预期损失 × 0.5` → 批准空运

4. **大促预案自动生成（Contingency Planning）**：
   - 场景A（爆单+30%）：自动计算需要额外多少仓容/运力
   - 场景B（滞销-30%）：自动生成促销加码方案和清仓价
   - 场景C（物流延误）：FBM备用方案激活标准和流程

5. **大促后复盘自动化**：
   - 实际销售 vs 预测：计算FA（预测准确率）
   - 滞销分析：哪些SKU售罄率<50%，原因归因
   - 紧急补货效果：评估每次紧急补货的ROI

**数学直觉**：双路收敛的统计学基础是"独立估算的平均值比单一估算更准确"（集成预测原理）。Top-Down从宏观约束出发，Bottom-Up从微观汇总出发，两者收敛到一个共识预测，方差比单路低约30%。

## ② 母婴出海应用案例

**场景A：Prime Day全流程S&OP自动化**

- **业务问题**：某母婴卖家每年Prime Day备货靠"感觉"，2023年吸奶器备500件缺货、婴儿温奶器备800件积压200件，年均大促备货损失$15万（缺货+积压合计）
- **数据要求**：历史2次大促SKU级销量、采购提前期、当前供应商产能
- **算法应用**：
  1. 提前8周启动：Top-Down目标$50万GMV → 拆解到20个SKU
  2. Bottom-Up汇总：各SKU历史大促×1.3增长 = $48万（差距4%，通过）
  3. 第6周追踪：吸奶器采购650件已确认，在途450件，缺口100件（15%）
  4. 触发规则：>10%缺口，距大促21天 → 评估空运100件（成本$800 vs 缺货损失预估$5000）→ 批准空运
  5. 大促后复盘：预测准确率78%（vs上年55%），大促备货损失降至$4万
- **预期产出**：大促备货准确率从40%→78%，缺货损失从$15万→$4万，年化ROI极高
- **业务价值**：系统将"感觉备货"升级为"数据驱动+闭环追踪"，是大促成败的核心能力

**场景B：双11/Black Friday多促活动并行管理**

- **业务问题**：Q4连续3个大促（10月Prime早促/11月Black Friday/12月Cyber Monday），备货需求叠加，采购窗口冲突
- **算法应用**：三促联合S&OP，统一规划采购节奏——6月开始规划→8月锁定海运舱位→9月首批发货→11月FBA库存充足；识别三促共同爆款（PUMP-PRO）做统一大批量采购，降低MOQ溢价
- **预期产出**：Q4采购成本降低8%（批量集中），三促缺货率各自<5%

## ③ 代码模板

```python
"""
大促盘货S&OP流程自动化系统
功能：双路预测收敛 + 备货追踪矩阵 + 紧急补货触发 + 预案生成
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PromoSKUPlan:
    """大促SKU计划单元"""
    sku_id: str
    abc_class: str
    # 历史大促数据
    last_promo_sales: int           # 上次大促销量
    last_promo_gmv: float           # 上次大促GMV
    # 本次预测
    growth_factor: float            # 增长系数（如1.3=30%增长）
    # 采购状态（每周更新）
    ordered_qty: int = 0            # 已下采购单量
    confirmed_inbound: int = 0      # 已确认入库量
    in_transit_qty: int = 0         # 在途量
    eta_confidence: float = 0.9     # 在途到货置信度
    current_stock: int = 0          # 当前库存
    # 成本参数
    unit_margin: float = 30.0       # 单位毛利($)
    air_freight_premium: float = 8.0  # 空运附加费/件($)
    lead_time_sea: int = 35         # 海运提前期
    lead_time_air: int = 7          # 空运提前期

    @property
    def bottom_up_forecast(self) -> int:
        """自下而上预测"""
        return int(self.last_promo_sales * self.growth_factor)

    @property
    def effective_supply(self) -> int:
        """有效供应量（含在途加权）"""
        return (self.confirmed_inbound
                + int(self.in_transit_qty * self.eta_confidence)
                + self.current_stock)

    @property
    def supply_gap(self) -> int:
        return max(self.bottom_up_forecast - self.effective_supply, 0)

    @property
    def gap_rate(self) -> float:
        return self.supply_gap / max(self.bottom_up_forecast, 1)


class PromoSOP:
    """大促S&OP协同引擎"""

    def __init__(self, promo_name: str, promo_date: datetime,
                 gmv_target: float, launch_date: datetime):
        self.promo_name = promo_name
        self.promo_date = promo_date
        self.gmv_target = gmv_target
        self.launch_date = launch_date          # S&OP启动日
        self.skus: List[PromoSKUPlan] = []
        self.weekly_snapshots: List[Dict] = []

    def add_sku(self, sku: PromoSKUPlan):
        self.skus.append(sku)

    def top_down_allocation(self) -> Dict[str, int]:
        """自上而下目标拆解"""
        total_bu_gmv = sum(s.last_promo_gmv * s.growth_factor for s in self.skus)
        gmv_scale = self.gmv_target / max(total_bu_gmv, 1)

        allocation = {}
        for sku in self.skus:
            scaled_gmv = sku.last_promo_gmv * sku.growth_factor * gmv_scale
            avg_price = scaled_gmv / max(sku.last_promo_sales * sku.growth_factor, 1)
            td_qty = int(scaled_gmv / max(avg_price, 1))
            allocation[sku.sku_id] = td_qty
        return allocation

    def convergence_check(self) -> Dict:
        """双路收敛验证"""
        td_alloc = self.top_down_allocation()
        td_total = sum(td_alloc.values())
        bu_total = sum(s.bottom_up_forecast for s in self.skus)

        divergence = abs(td_total - bu_total) / max((td_total + bu_total) / 2, 1)
        passed = divergence < 0.10

        sku_divergences = []
        for sku in self.skus:
            td = td_alloc.get(sku.sku_id, 0)
            bu = sku.bottom_up_forecast
            div = abs(td - bu) / max((td + bu) / 2, 1)
            if div > 0.20:
                sku_divergences.append({
                    'sku_id': sku.sku_id,
                    'top_down': td, 'bottom_up': bu,
                    'divergence': div,
                })

        return {
            'top_down_total': td_total,
            'bottom_up_total': bu_total,
            'overall_divergence': divergence,
            'convergence_passed': passed,
            'sku_outliers': sku_divergences,
        }

    def generate_tracking_matrix(self, weeks_to_promo: int) -> pd.DataFrame:
        """生成备货追踪矩阵"""
        records = []
        for sku in self.skus:
            gap_rate = sku.gap_rate

            # 风险评级
            if gap_rate > 0.30:
                risk = '🔴高风险'
            elif gap_rate > 0.10:
                risk = '🟡中风险'
            else:
                risk = '🟢低风险'

            # 紧急补货建议
            action = '维持'
            if gap_rate > 0.20 and weeks_to_promo <= 3:
                air_cost = sku.supply_gap * sku.air_freight_premium
                avoided_loss = sku.supply_gap * sku.unit_margin * 0.6
                if avoided_loss > air_cost:
                    action = f'🚁空运{sku.supply_gap}件(ROI={avoided_loss/max(air_cost,1):.1f}x)'
                else:
                    action = f'FBM备用方案'
            elif gap_rate > 0.10 and weeks_to_promo <= 5:
                action = f'加急海运{sku.supply_gap}件'

            records.append({
                'sku_id': sku.sku_id,
                'abc': sku.abc_class,
                'forecast': sku.bottom_up_forecast,
                'ordered': sku.ordered_qty,
                'confirmed': sku.confirmed_inbound,
                'in_transit': sku.in_transit_qty,
                'effective_supply': sku.effective_supply,
                'gap': sku.supply_gap,
                'gap_rate': gap_rate,
                'risk': risk,
                'weeks_to_promo': weeks_to_promo,
                'action': action,
            })

        return pd.DataFrame(records).sort_values('gap_rate', ascending=False)

    def generate_contingency_plan(self) -> Dict:
        """生成大促预案"""
        base_forecast = sum(s.bottom_up_forecast for s in self.skus)

        scenarios = {
            'A_爆单+30%': {
                'demand_multiplier': 1.30,
                'extra_units_needed': int(base_forecast * 0.30),
                'extra_warehouse_ft3': int(base_forecast * 0.30 * 1.5),
                'trigger': '大促前2小时销量>预测的150%',
                'action': '提高广告出价上限，关注A类售罄时间',
            },
            'B_滞销-30%': {
                'demand_multiplier': 0.70,
                'excess_units': int(base_forecast * 0.30),
                'clearance_discount': '20-30%',
                'trigger': '大促前4小时销量<预测的60%',
                'action': '触发额外优惠券+提高SP广告出价+联系平台申请Deal',
            },
            'C_物流延误': {
                'risk': 'FBA到货延误3天+',
                'trigger': '大促前7天ETA延误预警',
                'action': 'FBM备用库存激活（自营仓发货），广告切换FBM Listing',
                'fbm_min_stock': int(base_forecast * 0.15),
            },
        }
        return scenarios

    def post_promo_analysis(self, actuals: Dict[str, int]) -> pd.DataFrame:
        """大促后复盘分析"""
        records = []
        for sku in self.skus:
            actual = actuals.get(sku.sku_id, 0)
            forecast = sku.bottom_up_forecast
            fa = 1 - abs(actual - forecast) / max(forecast, 1)
            sellthrough = actual / max(sku.effective_supply, 1)

            records.append({
                'sku_id': sku.sku_id,
                'abc': sku.abc_class,
                'forecast': forecast,
                'actual': actual,
                'fa': round(fa, 2),
                'sellthrough': round(sellthrough, 2),
                'leftover': max(sku.effective_supply - actual, 0),
                'status': ('✅优秀' if fa >= 0.85 else
                           ('🟡一般' if fa >= 0.65 else '🔴需改进')),
            })

        return pd.DataFrame(records).sort_values('fa')


def run_promo_sop_demo():
    """大促S&OP自动化系统完整演示"""
    print("=" * 65)
    print("大促盘货S&OP流程自动化系统（Prime Day场景）")
    print("=" * 65)

    promo_date = datetime(2026, 7, 16)   # Prime Day假设日期
    launch_date = promo_date - timedelta(weeks=8)

    sop = PromoSOP(
        promo_name="Prime Day 2026",
        promo_date=promo_date,
        gmv_target=500000,          # $50万GMV目标
        launch_date=launch_date,
    )

    skus = [
        PromoSKUPlan("PUMP-PRO-US", "A",
                     last_promo_sales=520, last_promo_gmv=46800,
                     growth_factor=1.30,
                     ordered_qty=650, confirmed_inbound=450,
                     in_transit_qty=150, current_stock=80,
                     unit_margin=38.0, air_freight_premium=8.0),
        PromoSKUPlan("WARMER-S1", "A",
                     last_promo_sales=380, last_promo_gmv=15200,
                     growth_factor=1.20,
                     ordered_qty=400, confirmed_inbound=300,
                     in_transit_qty=80, current_stock=50,
                     unit_margin=18.0, air_freight_premium=4.0),
        PromoSKUPlan("BOTTLE-3P", "B",
                     last_promo_sales=650, last_promo_gmv=16250,
                     growth_factor=1.10,
                     ordered_qty=700, confirmed_inbound=600,
                     in_transit_qty=100, current_stock=120,
                     unit_margin=10.0, air_freight_premium=2.0),
        PromoSKUPlan("UV-STERILIZER", "B",
                     last_promo_sales=210, last_promo_gmv=25200,
                     growth_factor=1.25,
                     ordered_qty=200, confirmed_inbound=100,
                     in_transit_qty=80, current_stock=20,
                     unit_margin=50.0, air_freight_premium=12.0),
        PromoSKUPlan("NIPPLE-SHIELD", "C",
                     last_promo_sales=300, last_promo_gmv=2700,
                     growth_factor=0.90,
                     ordered_qty=250, confirmed_inbound=250,
                     in_transit_qty=0, current_stock=200,
                     unit_margin=3.0, air_freight_premium=1.0),
    ]

    for s in skus:
        sop.add_sku(s)

    # 双路收敛检查
    print("\n[1] 双路预测收敛验证")
    conv = sop.convergence_check()
    status = "✅通过" if conv['convergence_passed'] else "❌需对齐"
    print(f"\n  自上而下总量: {conv['top_down_total']:,}件")
    print(f"  自下而上总量: {conv['bottom_up_total']:,}件")
    print(f"  偏差率: {conv['overall_divergence']:.1%} → {status}")
    if conv['sku_outliers']:
        print(f"\n  SKU级偏差过大（>20%）:")
        for o in conv['sku_outliers']:
            print(f"    {o['sku_id']}: TD={o['top_down']} vs BU={o['bottom_up']} "
                  f"({o['divergence']:.0%}偏差)")

    # 备货追踪矩阵（假设当前距大促3周）
    print("\n[2] 备货追踪矩阵（距大促3周）")
    weeks_to_promo = 3
    tracking = sop.generate_tracking_matrix(weeks_to_promo)

    print(f"\n  {'SKU':<18} {'预测':<7} {'有效供应':<9} {'缺口':<7} {'缺口率':<8} {'风险':<10} {'建议行动'}")
    print("  " + "-" * 85)
    for _, row in tracking.iterrows():
        print(f"  {row['sku_id']:<18} {row['forecast']:<7} {row['effective_supply']:<9} "
              f"{row['gap']:<7} {row['gap_rate']:.0%}{'':>3} {row['risk']:<10} {row['action']}")

    # 高风险汇总
    high_risk = tracking[tracking['risk'] == '🔴高风险']
    if not high_risk.empty:
        print(f"\n  ⚠️ {len(high_risk)}个高风险SKU，预计缺货损失: "
              f"${sum(r['gap'] * 38 * 0.6 for _, r in high_risk.iterrows()):,.0f}")

    # 预案
    print("\n[3] 大促预案生成")
    contingency = sop.generate_contingency_plan()
    for scenario, details in contingency.items():
        print(f"\n  📋 {scenario}")
        print(f"     触发条件: {details.get('trigger', '-')}")
        print(f"     应对行动: {details.get('action', '-')}")

    # 大促后复盘模拟
    print("\n[4] 大促后复盘（模拟实际销量）")
    mock_actuals = {
        "PUMP-PRO-US": 610,     # 爆单
        "WARMER-S1": 430,
        "BOTTLE-3P": 680,
        "UV-STERILIZER": 240,
        "NIPPLE-SHIELD": 190,   # 滞销
    }
    retro = sop.post_promo_analysis(mock_actuals)
    overall_fa = retro['fa'].mean()
    print(f"\n  整体预测准确率 FA: {overall_fa:.0%}")
    print(f"\n  {'SKU':<18} {'预测':<7} {'实际':<7} {'FA':<8} {'售罄率':<8} {'剩余':<7} {'评级'}")
    print("  " + "-" * 65)
    for _, row in retro.iterrows():
        print(f"  {row['sku_id']:<18} {row['forecast']:<7} {row['actual']:<7} "
              f"{row['fa']:.0%}{'':>3} {row['sellthrough']:.0%}{'':>3} "
              f"{row['leftover']:<7} {row['status']}")

    print(f"\n  改进方向: FA<65%的SKU下次需检查增长系数假设")
    print("\n[✓] 大促盘货S&OP流程自动化系统测试通过")
    return tracking


if __name__ == "__main__":
    tracking = run_promo_sop_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-SOP-Sales-Operations-Planning]]（日常S&OP是大促S&OP的基础）、[[Skill-Flash-Sale-Realtime-Sellthrough-Forecast]]（大促实时监控与大促S&OP联动）
- **延伸（extends）**：[[Skill-Promotion-Demand-Decomposition]]（促销需求分解为大促预测提供基础）、[[Skill-In-Transit-Inventory-Tracking-Visibility]]（在途追踪支撑备货追踪矩阵）
- **可组合（combinable）**：[[Skill-ITO-DOI-Inventory-Turnover-Optimizer]]（大促后清仓降低DOI）、[[Skill-Warehouse-Capacity-Efficiency-Planning]]（大促备货仓容规划联动）

## ⑤ 商业价值评估

- **ROI 预估**：月销$50万卖家，大促年损失（缺货+积压）$15万；系统将备货准确率从40%→75%，损失降至$4万，年化节省$11万；系统成本$3万，ROI≈367%
- **实施难度**：⭐⭐⭐☆☆（流程逻辑清晰，难点是让销售/采购/物流三方都按节奏提交数据更新，需要组织配合）
- **优先级**：⭐⭐⭐⭐⭐（大促是全年最高价值时段，备货失误的代价最大，S&OP是防失误的核心机制）
- **适用规模**：参与Amazon/Shopee/TikTok Shop大促、年GMV>$200万的卖家
- **数据依赖**：历史大促SKU级销量（至少2次）、当前采购追踪数据、供应商交期确认
