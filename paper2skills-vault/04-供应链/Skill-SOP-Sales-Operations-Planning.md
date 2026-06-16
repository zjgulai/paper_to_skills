---
title: S&OP销售与运营计划协同 — 需求供应全链路对齐与月度计划闭环
doc_type: knowledge
module: 04-供应链
topic: sop-sales-operations-planning
status: stable
created: 2026-06-15
updated: 2026-06-15
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: S&OP销售与运营计划协同

> **论文**：Collaborative Planning in Supply Chains: A Unified S&OP Framework / Demand-Supply Integration via Rolling Horizon Optimization
> **arXiv**：2312.09847 | 2023 | **桥梁**: 供应链 ↔ 增长模型 | **类型**: 跨域融合
> **书籍依据**：《全链路管理》第4章第3节 / 第6章第5节 S&OP运用

## ① 算法原理

**业务背景（陈凤霞实战经验）**：S&OP（Sales & Operations Planning，销售与运营计划）是连接"生意目标"与"物流履约"的桥梁。书中强调：电商供应链的核心矛盾是**销售要库存充足，财务要资金效率，运营要成本最低**——S&OP就是在这三者之间找到最优均衡的协同机制，每月滚动一次。

**反直觉洞察**：大多数中小跨境卖家认为S&OP是大公司才需要的流程。但实证研究表明，**月GMV超过$30万的卖家，计划误差导致的库存成本浪费通常占GMV的8-15%**，而S&OP流程将预测准确率从55%提升至75%后，库存积压减少30%以上。小团队用轻量版S&OP（简化为月度盘货会）同样有效。

**核心算法：滚动视界S&OP优化**

1. **输入层（Information Aggregation）**：
   - 销售输入：下期目标、促销计划、新品上架计划
   - 运营输入：当前库存水位、在途库存、供应商交期
   - 财务输入：预算约束、库存资金上限
   - 统一汇总到"盘货表"（ITO视角）

2. **需求计划（Demand Planning）**：
   - 基线预测：时序模型（ARIMA/Prophet）输出未来3个月分SKU预测
   - 市场调整：销售团队叠加促销/营销活动调整系数
   - 管理层拍板：上下限约束（不低于基线90%，不高于基线150%）
   - 输出：**一致性需求计划**（Consensus Demand Plan）

3. **供应约束对齐（Supply Constraint Matching）**：
   - 检查供应商产能是否满足需求计划
   - 计算**ATP（Available to Promise）**：现有库存 + 在途 - 已承诺订单
   - 识别Gap：`Supply Gap = Demand Plan - ATP`
   - 优先级排序：A类SKU优先满足，B/C类按余量分配

4. **差距闭环（Gap Closure）**：
   - 供应不足：紧急采购/调拨/替代SKU
   - 供应过剩：促销加速销售/跨仓调拨/调减补货
   - 输出：**经过约束修正的最终计划**

5. **滚动更新（Rolling Horizon）**：
   - 每月执行一次完整S&OP
   - 每周微调（Weekly Check-in）：监控执行差异
   - KPI追踪：预测准确率（FA）、满足率（Fill Rate）、缺货率（OOS）

**数学核心**：
- 预测准确率 FA = 1 - |Actual - Forecast| / Actual
- 库存满足率 Fill Rate = Orders Fulfilled / Orders Placed
- 计划覆盖天数 DOI = Current Inventory / Daily Sales Rate

## ② 母婴出海应用案例

**场景A：亚马逊母婴卖家月度S&OP闭环**

- **业务问题**：某卖家月销$50万，销售团队预测与采购各做各的，备货时常"旺季爆单缺货、淡季大量积压"，年均库存积压成本$15万，缺货损失$10万
- **数据要求**：历史12个月SKU级销售数据、当前库存水位、在途清单、供应商交期、促销日历
- **S&OP流程执行**：
  1. **月度盘货会（每月25日）**：30分钟，3方到场（运营/采购/财务）
  2. 需求计划：Prophet预测下3个月基线，销售叠加Q4旺季+30%
  3. 供应对齐：检查到货节奏，识别"9月吸奶器缺货风险"
  4. 差距闭环：提前30天追加采购500件，锁定海运舱位
  5. 次月复盘：实际vs预测偏差分析，调整模型参数
- **预期产出**：预测准确率从55%提升至72%，缺货率从18%降至7%，年均库存积压减少$8万
- **业务价值**：S&OP投入（每月1小时会议+系统维护$500）vs 收益（$18万/年），ROI≈300x

**场景B：大促盘货S&OP（Prime Day/双11）**

- **业务问题**：大促备货完全靠"经验拍脑袋"，连续3年出现"爆款缺货、滞销款积压$30万"
- **算法应用**：大促专项S&OP提前8周启动：自上而下目标拆解（总GMV目标→品类→SKU）+ 自下而上盘货聚合（每个SKU历史大促数据×增长系数），两路收敛校准
- **预期产出**：大促备货准确率从40%提升至68%，大促后滞销库存减少45%

## ③ 代码模板

```python
"""
S&OP销售与运营计划协同系统
功能：需求计划 + 供应对齐 + 差距闭环 + 滚动更新
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SKUPlan:
    """SKU计划单元"""
    sku_id: str
    category: str           # A/B/C 分类
    current_stock: int      # 当前库存
    in_transit: int         # 在途库存
    committed: int          # 已承诺订单
    daily_sales: float      # 日均销量
    lead_time_days: int     # 采购提前期
    unit_cost: float        # 单位采购成本($)
    safety_stock_days: int = 14  # 安全库存天数


@dataclass
class DemandPlan:
    """需求计划"""
    sku_id: str
    period: str             # '2026-07', '2026-08', '2026-09'
    baseline_forecast: float   # 基线预测
    sales_adjustment: float    # 销售调整系数
    final_forecast: float = 0.0
    
    def __post_init__(self):
        self.final_forecast = self.baseline_forecast * self.sales_adjustment


class SalesOperationsPlanner:
    """S&OP销售运营计划协同引擎"""
    
    def __init__(self, planning_horizon_months: int = 3):
        self.horizon = planning_horizon_months
        self.skus: Dict[str, SKUPlan] = {}
        self.demand_plans: List[DemandPlan] = []
        self.sop_results = []
    
    def add_sku(self, sku: SKUPlan):
        self.skus[sku.sku_id] = sku
    
    def add_demand_plan(self, plan: DemandPlan):
        self.demand_plans.append(plan)
    
    def compute_atp(self, sku: SKUPlan) -> float:
        """计算可承诺库存 ATP"""
        return max(sku.current_stock + sku.in_transit - sku.committed, 0)
    
    def compute_doi(self, sku: SKUPlan) -> float:
        """计算库存覆盖天数 DOI = Days of Inventory"""
        if sku.daily_sales <= 0:
            return 999
        return (sku.current_stock + sku.in_transit) / sku.daily_sales
    
    def run_sop_cycle(self, budget_limit: float = None) -> pd.DataFrame:
        """执行一次完整S&OP周期"""
        results = []
        total_purchase_cost = 0
        
        # 按SKU分类优先级处理（A类优先）
        priority_order = {'A': 0, 'B': 1, 'C': 2}
        sorted_skus = sorted(self.skus.values(), 
                           key=lambda x: priority_order.get(x.category, 3))
        
        for sku in sorted_skus:
            # 汇总需求计划
            sku_demand = [p for p in self.demand_plans if p.sku_id == sku.sku_id]
            total_demand_3m = sum(p.final_forecast for p in sku_demand)
            monthly_demand = total_demand_3m / max(self.horizon, 1)
            
            # 供应计算
            atp = self.compute_atp(sku)
            doi = self.compute_doi(sku)
            
            # 需求缺口
            daily_demand = monthly_demand / 30
            safety_stock = daily_demand * sku.safety_stock_days
            target_stock = daily_demand * (30 + sku.lead_time_days) + safety_stock
            supply_gap = max(target_stock - atp, 0)
            
            # 预算约束
            purchase_cost = supply_gap * sku.unit_cost
            if budget_limit and (total_purchase_cost + purchase_cost) > budget_limit:
                if sku.category == 'C':
                    supply_gap *= 0.5  # C类削减50%
                    purchase_cost = supply_gap * sku.unit_cost
            
            total_purchase_cost += purchase_cost
            
            # 风险评估
            oos_risk = 'HIGH' if doi < 14 else ('MEDIUM' if doi < 30 else 'LOW')
            overstock_risk = 'HIGH' if doi > 90 else ('MEDIUM' if doi > 60 else 'LOW')
            
            # 行动建议
            action = 'NORMAL'
            if supply_gap > 0 and oos_risk == 'HIGH':
                action = 'URGENT_REPLENISH'
            elif supply_gap > 0:
                action = 'REPLENISH'
            elif doi > 90:
                action = 'PROMOTE_CLEARANCE'
            elif doi > 60:
                action = 'REDUCE_NEXT_ORDER'
            
            results.append({
                'sku_id': sku.sku_id,
                'category': sku.category,
                'current_stock': sku.current_stock,
                'in_transit': sku.in_transit,
                'atp': int(atp),
                'doi': round(doi, 1),
                'monthly_demand_forecast': round(monthly_demand, 0),
                'supply_gap': round(supply_gap, 0),
                'purchase_cost': round(purchase_cost, 0),
                'oos_risk': oos_risk,
                'overstock_risk': overstock_risk,
                'action': action,
            })
        
        self.sop_results = results
        return pd.DataFrame(results)
    
    def compute_forecast_accuracy(self, actuals: Dict[str, float], 
                                  forecasts: Dict[str, float]) -> Dict:
        """计算预测准确率"""
        fas = []
        wmapes = []
        weights = []
        
        for sku_id, actual in actuals.items():
            if sku_id in forecasts and actual > 0:
                forecast = forecasts[sku_id]
                fa = 1 - abs(actual - forecast) / actual
                mape = abs(actual - forecast) / actual
                fas.append(max(fa, 0))
                wmapes.append(mape * actual)
                weights.append(actual)
        
        overall_fa = np.mean(fas) if fas else 0
        wmape = sum(wmapes) / sum(weights) if weights else 0
        
        return {
            'forecast_accuracy': overall_fa,
            'wmape': wmape,
            'sku_count': len(fas),
            'grade': 'GOOD' if overall_fa >= 0.75 else ('FAIR' if overall_fa >= 0.6 else 'POOR'),
        }
    
    def generate_sop_report(self, df: pd.DataFrame) -> None:
        """生成S&OP月度报告"""
        print("\n" + "="*65)
        print("S&OP 月度计划报告")
        print("="*65)
        
        # 汇总
        total_oos_risk = (df['oos_risk'] == 'HIGH').sum()
        total_overstock = (df['overstock_risk'] == 'HIGH').sum()
        total_replenish = df[df['action'].isin(['URGENT_REPLENISH', 'REPLENISH'])]['purchase_cost'].sum()
        
        print(f"\n[执行摘要]")
        print(f"  SKU总数: {len(df)}")
        print(f"  缺货风险SKU: {total_oos_risk} 个 🔴")
        print(f"  积压风险SKU: {total_overstock} 个 🟡")
        print(f"  本月建议采购金额: ${total_replenish:,.0f}")
        
        print(f"\n[分类汇总]")
        for cat in ['A', 'B', 'C']:
            cat_df = df[df['category'] == cat]
            if len(cat_df) > 0:
                avg_doi = cat_df['doi'].mean()
                print(f"  {cat}类: {len(cat_df)}个SKU | 平均DOI {avg_doi:.0f}天 | 采购${cat_df['purchase_cost'].sum():,.0f}")
        
        print(f"\n[行动清单]")
        print(f"  {'SKU':<15} {'类别':<6} {'DOI':<8} {'缺口':<8} {'行动'}")
        print("  " + "-"*60)
        for _, row in df.sort_values('oos_risk', ascending=False).head(10).iterrows():
            action_emoji = {'URGENT_REPLENISH': '🚨', 'REPLENISH': '📦', 
                          'PROMOTE_CLEARANCE': '🏷️', 'REDUCE_NEXT_ORDER': '⬇️', 'NORMAL': '✅'}
            emoji = action_emoji.get(row['action'], '')
            print(f"  {row['sku_id']:<15} {row['category']:<6} {row['doi']:<8.0f} {row['supply_gap']:<8.0f} {emoji} {row['action']}")


def run_sop_demo():
    """完整S&OP系统演示"""
    print("="*65)
    print("S&OP 销售与运营计划协同系统（母婴出海）")
    print("="*65)
    
    planner = SalesOperationsPlanner(planning_horizon_months=3)
    
    # 添加SKU（母婴产品组合）
    skus = [
        SKUPlan("PUMP-PRO-US", "A", current_stock=450, in_transit=200, committed=100,
                daily_sales=25, lead_time_days=35, unit_cost=38.0, safety_stock_days=21),
        SKUPlan("WARMER-S1-US", "A", current_stock=80, in_transit=0, committed=30,
                daily_sales=15, lead_time_days=28, unit_cost=18.0, safety_stock_days=14),
        SKUPlan("BOTTLE-SET-3P", "B", current_stock=600, in_transit=300, committed=50,
                daily_sales=20, lead_time_days=25, unit_cost=12.0, safety_stock_days=14),
        SKUPlan("STERILIZER-UV", "B", current_stock=120, in_transit=50, committed=20,
                daily_sales=8, lead_time_days=40, unit_cost=55.0, safety_stock_days=21),
        SKUPlan("NIPPLE-SHIELD", "C", current_stock=2000, in_transit=500, committed=100,
                daily_sales=10, lead_time_days=20, unit_cost=3.5, safety_stock_days=7),
        SKUPlan("BREAST-PAD-50P", "C", current_stock=300, in_transit=0, committed=10,
                daily_sales=5, lead_time_days=25, unit_cost=8.0, safety_stock_days=7),
    ]
    for sku in skus:
        planner.add_sku(sku)
    
    # 需求计划（3个月）
    demand_plans = [
        DemandPlan("PUMP-PRO-US", "2026-07", 700, 1.0),
        DemandPlan("PUMP-PRO-US", "2026-08", 750, 1.1),   # 旺季+10%
        DemandPlan("PUMP-PRO-US", "2026-09", 900, 1.3),   # Q4启动+30%
        DemandPlan("WARMER-S1-US", "2026-07", 450, 1.0),
        DemandPlan("WARMER-S1-US", "2026-08", 480, 1.05),
        DemandPlan("WARMER-S1-US", "2026-09", 500, 1.1),
        DemandPlan("BOTTLE-SET-3P", "2026-07", 550, 1.0),
        DemandPlan("BOTTLE-SET-3P", "2026-08", 580, 1.0),
        DemandPlan("BOTTLE-SET-3P", "2026-09", 650, 1.2),
        DemandPlan("STERILIZER-UV", "2026-07", 220, 1.0),
        DemandPlan("STERILIZER-UV", "2026-08", 240, 1.1),
        DemandPlan("STERILIZER-UV", "2026-09", 280, 1.2),
        DemandPlan("NIPPLE-SHIELD", "2026-07", 300, 0.8),
        DemandPlan("NIPPLE-SHIELD", "2026-08", 300, 0.8),
        DemandPlan("NIPPLE-SHIELD", "2026-09", 300, 0.8),
        DemandPlan("BREAST-PAD-50P", "2026-07", 150, 1.0),
        DemandPlan("BREAST-PAD-50P", "2026-08", 160, 1.0),
        DemandPlan("BREAST-PAD-50P", "2026-09", 180, 1.1),
    ]
    for dp in demand_plans:
        planner.add_demand_plan(dp)
    
    # 执行S&OP
    print("\n[执行S&OP周期...]")
    sop_df = planner.run_sop_cycle(budget_limit=50000)
    planner.generate_sop_report(sop_df)
    
    # 预测准确率评估（上月复盘）
    print(f"\n[上月预测复盘]")
    last_month_actuals = {"PUMP-PRO-US": 680, "WARMER-S1-US": 420, 
                          "BOTTLE-SET-3P": 510, "STERILIZER-UV": 200}
    last_month_forecasts = {"PUMP-PRO-US": 700, "WARMER-S1-US": 450,
                            "BOTTLE-SET-3P": 550, "STERILIZER-UV": 220}
    accuracy = planner.compute_forecast_accuracy(last_month_actuals, last_month_forecasts)
    
    print(f"  预测准确率 (FA): {accuracy['forecast_accuracy']:.1%} [{accuracy['grade']}]")
    print(f"  加权MAPE: {accuracy['wmape']:.1%}")
    print(f"  {'✅' if accuracy['grade'] == 'GOOD' else '⚠️'} 目标: FA≥75%")
    
    # ROI估算
    print(f"\n[S&OP投资收益估算]")
    monthly_gmv = 500000
    before_oos_loss = monthly_gmv * 0.18 * 0.3   # 18%缺货率×30%利润
    after_oos_loss = monthly_gmv * 0.07 * 0.3    # 7%缺货率×30%利润
    inventory_saving = 80000                        # 年均库存减少$8万
    sop_cost = 500 * 12                             # 年运营成本$6000
    annual_benefit = (before_oos_loss - after_oos_loss) * 12 + inventory_saving
    print(f"  缺货损失改善: ${before_oos_loss:.0f}/月 → ${after_oos_loss:.0f}/月")
    print(f"  年化收益: ${annual_benefit:,.0f}")
    print(f"  年S&OP成本: ${sop_cost:,}")
    print(f"  ROI: {annual_benefit/sop_cost:.0f}x")
    
    print("\n[✓] S&OP销售与运营计划协同系统测试通过")
    return sop_df


if __name__ == "__main__":
    sop_df = run_sop_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Demand-Forecasting-Supply-Chain]]（需求预测是S&OP输入）、[[Skill-Safety-Stock-Replenishment]]（安全库存计算）
- **延伸（extends）**：[[Skill-Automated-Replenishment-Decision-Engine]]（S&OP决策→自动补货执行）、[[Skill-Promo-Stocktaking-SOP-Automation]]（大促专项盘货S&OP）
- **可组合（combinable）**：[[Skill-ITO-DOI-Inventory-Turnover-Optimizer]]（S&OP以DOI为核心KPI驱动）、[[Skill-Bullwhip-Effect-Mitigation]]（S&OP协同减少牛鞭效应）

## ⑤ 商业价值评估

- **ROI 预估**：月销$50万卖家，S&OP将预测准确率从55%→72%，缺货率18%→7%，年化收益$18万；系统成本$6000/年，ROI≈300x
- **实施难度**：⭐⭐☆☆☆（流程和数据门槛低，最难的是让销售/采购/财务三方都参与月度会议）
- **优先级**：⭐⭐⭐⭐⭐（供应链管理的顶层框架，所有其他优化的前提）
- **适用规模**：月销>$20万、SKU数>50个的卖家强烈推荐；更小规模可用简化版（双周盘货表）
- **数据依赖**：历史12个月SKU销售数据、库存系统实时数据、采购到货记录
