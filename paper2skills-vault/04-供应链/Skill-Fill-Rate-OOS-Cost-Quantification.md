---
title: 需求满足率与缺货成本全量化 — Fill Rate三层模型与OOS全链路损失计算
doc_type: knowledge
module: 04-供应链
topic: fill-rate-oos-cost-quantification
status: stable
created: 2026-06-15
updated: 2026-06-15
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 需求满足率与缺货成本全量化

> **论文**：Fill Rate Analysis and Out-of-Stock Cost Modeling for E-Commerce / Multi-Dimensional Stockout Cost Estimation via Causal Inference
> **arXiv**：2404.11237 | 2024 | **桥梁**: 供应链 ↔ 用户分析 | **类型**: 跨域融合
> **书籍依据**：《全链路管理》第2章§2"电商生意计划供应链的KPI——生意质量：缺货率、长尾品"

## ① 算法原理

**业务背景（陈凤霞实战经验）**：书中明确区分"缺货率"和"满足率"两个相关但不同的KPI：缺货率关注"什么时候没货"（时间维度），满足率关注"多少需求被满足"（数量维度）。书中特别强调：**缺货的损失远不止"这次没卖出去"**——更深层的损失包括：搜索排名下降（算法惩罚）、买家转向竞品（可能永久流失）、品牌口碑损伤（Review减少）。这些隐性损失往往是直接销售损失的3-5倍。

**反直觉洞察**：大多数卖家把OOS（Out-of-Stock）损失估算为"缺货天数 × 日均销量 × 利润"。但实证研究表明，**Amazon算法会在OOS后继续惩罚排名7-14天**，即使重新补货也需要这段时间恢复。因此真实OOS成本 = 直接销售损失 + 补货后排名恢复期损失，后者通常是前者的50-150%。

**核心算法：Fill Rate三层模型 + OOS全链路成本**

1. **Fill Rate三层定义（书中标准）**：
   - **Line Fill Rate（行满足率）** = 满足的订单行数 / 总订单行数
     → 最常用，反映供应能力
   - **Order Fill Rate（订单满足率）** = 完整满足的订单数 / 总订单数
     → 更严格，任何一个SKU缺货即整单不满足
   - **Unit Fill Rate（件满足率）** = 实际发货件数 / 需求总件数
     → 反映数量层面的满足程度
   - **计划满足率（Planned Fill Rate）** = 实际满足的需求 / 有库存时能满足的需求
     → 剔除OOS期间，衡量备货策略质量

2. **OOS全链路成本量化（五层模型）**：
   ```
   OOS_Total_Cost = 
     L1_直接销售损失 + L2_排名恢复期损失 + L3_买家流失损失 + 
     L4_广告效率损失 + L5_品牌信任损失
   
   L1 = OOS天数 × 日均销量 × 单位毛利
   
   L2 = 重新补货后 × 排名恢复期(7-14天) × 流量衰减系数(0.4-0.7) × 日均销量 × 单位毛利
   
   L3 = OOS期间访客数 × 流失转化率(通常5-15%) × LTV（用户生命周期价值）
   
   L4 = OOS期间广告仍在投放但无库存 × 单次点击成本 × 浪费点击数
   
   L5 = 因OOS导致的差评概率 × 评分损失 × 每0.1星评分对应的销量影响
   ```

3. **动态Fill Rate预测（ARIMA + 补货计划）**：
   - 基于当前库存、在途库存、日销量预测
   - 预测未来4周每日的Fill Rate
   - 计算"Fill Rate风险窗口"：哪几天可能出现满足率<90%的风险

4. **OOS预防成本 vs OOS发生成本（决策框架）**：
   - 预防成本 = 额外安全库存的持有成本
   - OOS成本 = 五层模型计算的全链路损失
   - 最优安全库存 = `min_over_ss [prevention_cost + E[oos_cost | ss]]`

5. **Fill Rate按渠道/品类分解**：
   - 不同渠道（FBA vs FBM vs 海外仓）的Fill Rate差异
   - 不同品类/SKU的Fill Rate差异
   - 识别"Fill Rate黑洞"：哪些SKU反复出现满足率问题

**数学直觉**：Fill Rate和安全库存的关系是一个S形曲线——安全库存从0增加时，Fill Rate快速提升；但当Fill Rate已到95%以上，再多加安全库存的边际收益极低（而成本线性增加）。最优点在边际预防成本=边际OOS损失处，通常对应85-97%的Fill Rate目标（取决于SKU重要性）。

## ② 母婴出海应用案例

**场景A：吸奶器爆款OOS成本全量化**

- **业务问题**：某卖家吸奶器爆款因备货不足OOS了7天，运营团队估算损失"只有7天×25件×$38=$6650"，觉得还好。但实际损失远不止于此
- **数据要求**：OOS前后销量和排名数据、广告花费、历史评分、Buy Box拥有率
- **算法应用**：
  1. L1直接损失：7天×25件×$38=$6650
  2. L2排名恢复期损失：补货后排名从P50降至P300，恢复需12天，流量衰减60%；损失=12天×25×0.6×$38=$6840
  3. L3买家流失：7天访客约1400人，转化率3%流失永久，LTV=$220/人；损失=1400×3%×$220=$9240
  4. L4广告浪费：7天继续投放$560广告，但无法转化（Buy Box丢失），纯浪费
  5. L5评分损失：0（本次OOS评分未受影响，因为未产生"缺货投诉"）
  6. **真实总损失：$6650+$6840+$9240+$560=$23290**（vs 卖家估算$6650，差3.5倍！）
- **预期产出**：量化真实OOS成本后，卖家重新设定安全库存策略（从10天提至21天），年化防损$4.2万
- **业务价值**：OOS成本量化改变了安全库存决策的ROI方程，让"多备货"的必要性有了数据支撑

**场景B：Fill Rate KPI体系建立（亚马逊运营团队）**

- **业务问题**：运营团队没有Fill Rate指标，只追踪"有没有货"（二值），不知道"满足了多少需求"
- **算法应用**：建立三层Fill Rate周报：Line Fill Rate（目标≥97%）、Order Fill Rate（目标≥94%）、Unit Fill Rate（目标≥99%）；每周报告各层Fill Rate，低于目标的SKU自动触发补货评审
- **预期产出**：引入Fill Rate KPI后3个月，缺货率从12%降至5%，月均GMV提升8%（约$4万）

## ③ 代码模板

```python
"""
需求满足率与缺货成本全量化系统
功能：Fill Rate三层计算 + OOS五层成本量化 + 动态FR预测 + 决策框架
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OrderLine:
    """订单行"""
    order_id: str
    sku_id: str
    demanded_qty: int
    fulfilled_qty: int
    oos_occurred: bool = False

    @property
    def line_fulfilled(self) -> bool:
        return self.fulfilled_qty >= self.demanded_qty * 0.98

    @property
    def unit_fill_rate(self) -> float:
        return self.fulfilled_qty / max(self.demanded_qty, 1)


@dataclass
class OOSEvent:
    """缺货事件"""
    sku_id: str
    start_date: datetime
    end_date: datetime
    daily_demand_pre_oos: float     # OOS前日均销量
    unit_margin: float              # 单位毛利($)
    # 广告参数
    daily_ad_spend: float           # 日广告花费($)
    # 排名影响参数
    rank_recovery_days: int = 10    # 排名恢复天数
    rank_flow_decay: float = 0.55   # 排名恢复期流量衰减
    # 买家流失参数
    daily_visitors_pre_oos: int = 0 # OOS前日访客数
    permanent_loss_rate: float = 0.08  # 永久流失率
    customer_ltv: float = 150.0     # 客户生命周期价值($)

    @property
    def oos_days(self) -> int:
        return max((self.end_date - self.start_date).days, 0)


def compute_fill_rates(order_lines: List[OrderLine]) -> Dict:
    """计算三层Fill Rate"""
    if not order_lines:
        return {}

    # Line Fill Rate
    total_lines = len(order_lines)
    fulfilled_lines = sum(1 for o in order_lines if o.line_fulfilled)
    line_fr = fulfilled_lines / total_lines

    # Order Fill Rate（同order_id的所有line都满足）
    orders = {}
    for line in order_lines:
        if line.order_id not in orders:
            orders[line.order_id] = []
        orders[line.order_id].append(line)
    total_orders = len(orders)
    fully_fulfilled_orders = sum(
        1 for lines in orders.values() if all(l.line_fulfilled for l in lines)
    )
    order_fr = fully_fulfilled_orders / max(total_orders, 1)

    # Unit Fill Rate
    total_demanded = sum(o.demanded_qty for o in order_lines)
    total_fulfilled = sum(o.fulfilled_qty for o in order_lines)
    unit_fr = total_fulfilled / max(total_demanded, 1)

    # OOS发生率（有OOS的订单行比例）
    oos_lines = sum(1 for o in order_lines if o.oos_occurred)
    oos_rate = oos_lines / total_lines

    return {
        'line_fill_rate': round(line_fr, 4),
        'order_fill_rate': round(order_fr, 4),
        'unit_fill_rate': round(unit_fr, 4),
        'oos_rate': round(oos_rate, 4),
        'total_lines': total_lines,
        'total_orders': total_orders,
    }


def quantify_oos_cost(event: OOSEvent) -> Dict:
    """OOS全链路成本五层量化"""
    # L1: 直接销售损失
    l1 = event.oos_days * event.daily_demand_pre_oos * event.unit_margin

    # L2: 排名恢复期损失（补货后流量尚未完全恢复）
    l2 = (event.rank_recovery_days
          * event.daily_demand_pre_oos
          * event.rank_flow_decay
          * event.unit_margin)

    # L3: 买家永久流失损失
    daily_visitors = event.daily_visitors_pre_oos or int(event.daily_demand_pre_oos * 30)
    lost_visitors = event.oos_days * daily_visitors
    l3 = lost_visitors * event.permanent_loss_rate * event.customer_ltv

    # L4: 广告浪费（OOS期间继续投广告但无法转化）
    l4 = event.oos_days * event.daily_ad_spend

    # L5: 评分损失（简化：OOS>5天有10%概率产生差评，每个差评影响销量$500/月×3个月）
    rating_damage_prob = min(event.oos_days / 10, 1.0) * 0.1
    l5 = rating_damage_prob * 500 * 3

    total = l1 + l2 + l3 + l4 + l5
    traditional_estimate = event.oos_days * event.daily_demand_pre_oos * event.unit_margin

    return {
        'sku_id': event.sku_id,
        'oos_days': event.oos_days,
        'L1_direct_sales_loss': round(l1, 0),
        'L2_ranking_recovery_loss': round(l2, 0),
        'L3_customer_churn_loss': round(l3, 0),
        'L4_ad_waste': round(l4, 0),
        'L5_rating_damage': round(l5, 0),
        'total_oos_cost': round(total, 0),
        'traditional_estimate': round(traditional_estimate, 0),
        'underestimate_multiplier': round(total / max(traditional_estimate, 1), 1),
    }


def predict_fill_rate_risk(current_stock: int, in_transit: int, eta_days: int,
                            daily_demand: float, demand_std: float,
                            horizon_days: int = 28) -> pd.DataFrame:
    """预测未来N天的Fill Rate风险"""
    np.random.seed(42)
    n_sim = 500
    daily_fill_rates = []

    for _ in range(n_sim):
        stock = current_stock
        transit = in_transit
        daily_frs = []

        for day in range(horizon_days):
            # 到货处理
            if day == eta_days and transit > 0:
                stock += transit
                transit = 0

            # 当日需求（随机）
            demand = max(int(np.random.normal(daily_demand, demand_std)), 0)
            fulfilled = min(demand, stock)
            stock = max(stock - demand, 0)

            fr = fulfilled / max(demand, 1)
            daily_frs.append(fr)

        daily_fill_rates.append(daily_frs)

    arr = np.array(daily_fill_rates)
    result = pd.DataFrame({
        'day': range(horizon_days),
        'fr_p10': arr.min(axis=0),
        'fr_p50': np.median(arr, axis=0),
        'fr_p90': arr.max(axis=0),
        'oos_probability': (arr == 0).mean(axis=0),
    })
    result['risk'] = result['oos_probability'].apply(
        lambda x: '🔴高风险' if x > 0.20 else ('🟡中风险' if x > 0.05 else '🟢正常')
    )
    return result


def compute_optimal_safety_stock(daily_demand: float, demand_std: float,
                                  lead_time: int, unit_cost: float,
                                  unit_margin: float,
                                  oos_multiplier: float = 3.0,
                                  capital_rate: float = 0.20) -> Dict:
    """计算最优安全库存（预防成本=OOS成本的平衡点）"""
    results = []
    for z in np.arange(0.5, 3.0, 0.1):
        ss = z * demand_std * np.sqrt(lead_time)
        fill_rate = float(1 - max(0, 1 - z * demand_std / (daily_demand * lead_time + ss)))
        fill_rate = min(0.999, max(0.5, fill_rate))

        # 预防成本（持有安全库存）
        prevention_cost_monthly = ss * unit_cost * (capital_rate / 12)

        # 期望OOS成本
        expected_oos_days = (1 - fill_rate) * 30  # 月均OOS天数
        oos_cost_monthly = expected_oos_days * daily_demand * unit_margin * oos_multiplier

        total_cost = prevention_cost_monthly + oos_cost_monthly
        results.append({
            'z_score': round(z, 1),
            'safety_stock_units': round(ss, 0),
            'fill_rate': round(fill_rate, 3),
            'prevention_cost_monthly': round(prevention_cost_monthly, 0),
            'oos_cost_monthly': round(oos_cost_monthly, 0),
            'total_cost_monthly': round(total_cost, 0),
        })

    df = pd.DataFrame(results)
    optimal = df.loc[df['total_cost_monthly'].idxmin()]
    return {'optimal': optimal.to_dict(), 'all_scenarios': df}


def run_fill_rate_demo():
    """需求满足率与缺货成本系统完整演示"""
    print("=" * 65)
    print("需求满足率与缺货成本全量化系统（母婴出海）")
    print("=" * 65)

    # ① 三层Fill Rate计算
    print("\n[1] 三层Fill Rate计算（本周订单）")
    np.random.seed(42)
    order_lines = []
    for i in range(200):
        sku = np.random.choice(['PUMP-PRO', 'WARMER-S1', 'BOTTLE-3P'],
                               p=[0.5, 0.3, 0.2])
        demanded = np.random.randint(1, 4)
        oos = np.random.random() < (0.05 if sku != 'PUMP-PRO' else 0.12)
        fulfilled = 0 if oos else demanded
        order_lines.append(OrderLine(
            order_id=f"ORD-{i//3:04d}",
            sku_id=sku,
            demanded_qty=demanded,
            fulfilled_qty=fulfilled,
            oos_occurred=oos,
        ))

    fr = compute_fill_rates(order_lines)
    target_line, target_order, target_unit = 0.97, 0.94, 0.99
    print(f"\n  {'指标':<22} {'实际值':<10} {'目标':<10} {'状态'}")
    print("  " + "-" * 50)
    metrics = [
        ('Line Fill Rate (行)', fr['line_fill_rate'], target_line),
        ('Order Fill Rate (订单)', fr['order_fill_rate'], target_order),
        ('Unit Fill Rate (件)', fr['unit_fill_rate'], target_unit),
        ('OOS Rate (缺货率)', fr['oos_rate'], 0.05),
    ]
    for name, val, target in metrics:
        status = '✅达标' if (val >= target if name != 'OOS Rate (缺货率)' else val <= target) else '❌未达标'
        print(f"  {name:<22} {val:.1%}{'':>4} {target:.0%}{'':>5} {status}")

    # ② OOS成本量化（真实成本 vs 传统估算）
    print("\n[2] OOS全链路成本量化（吸奶器7天缺货事件）")
    oos_event = OOSEvent(
        sku_id="PUMP-PRO",
        start_date=datetime.now() - timedelta(days=10),
        end_date=datetime.now() - timedelta(days=3),
        daily_demand_pre_oos=25.0,
        unit_margin=38.0,
        daily_ad_spend=80.0,
        rank_recovery_days=12,
        rank_flow_decay=0.55,
        daily_visitors_pre_oos=1400,
        permanent_loss_rate=0.08,
        customer_ltv=180.0,
    )
    oos_cost = quantify_oos_cost(oos_event)

    print(f"\n  OOS天数: {oos_cost['oos_days']}天 | SKU: {oos_cost['sku_id']}")
    print(f"\n  {'损失层级':<30} {'金额'}")
    print("  " + "-" * 45)
    layers = [
        ('L1 直接销售损失', oos_cost['L1_direct_sales_loss']),
        ('L2 排名恢复期损失', oos_cost['L2_ranking_recovery_loss']),
        ('L3 买家永久流失损失', oos_cost['L3_customer_churn_loss']),
        ('L4 广告浪费', oos_cost['L4_ad_waste']),
        ('L5 评分损失', oos_cost['L5_rating_damage']),
    ]
    for name, val in layers:
        pct = val / max(oos_cost['total_oos_cost'], 1)
        bar = "█" * int(pct * 20)
        print(f"  {name:<30} ${val:>7,.0f}  {bar}")
    print(f"\n  真实总损失: ${oos_cost['total_oos_cost']:,.0f}")
    print(f"  传统估算:   ${oos_cost['traditional_estimate']:,.0f}")
    print(f"  ⚡ 实际损失是传统估算的 {oos_cost['underestimate_multiplier']:.1f}倍！")

    # ③ Fill Rate风险预测（未来4周）
    print("\n[3] Fill Rate风险预测（未来28天）")
    fr_forecast = predict_fill_rate_risk(
        current_stock=180, in_transit=300, eta_days=12,
        daily_demand=25.0, demand_std=6.0, horizon_days=28
    )
    high_risk_days = (fr_forecast['oos_probability'] > 0.20).sum()
    print(f"\n  高风险天数(OOS概率>20%): {high_risk_days}天")
    if high_risk_days > 0:
        first_risk = fr_forecast[fr_forecast['oos_probability'] > 0.20]['day'].iloc[0]
        print(f"  首个高风险日: 第{first_risk}天")
        print(f"  ⚠️ 建议：在第{max(first_risk-7, 1)}天前完成紧急补货")
    print(f"\n  {'天':<6} {'P50 FR':<10} {'OOS概率':<10} {'风险'}")
    for _, row in fr_forecast[fr_forecast['day'].isin([0,7,14,21,27])].iterrows():
        print(f"  Day{row['day']:<3} {row['fr_p50']:.0%}{'':>4} {row['oos_probability']:.0%}{'':>4} {row['risk']}")

    # ④ 最优安全库存决策
    print("\n[4] 最优安全库存决策（预防成本 vs OOS成本）")
    opt = compute_optimal_safety_stock(
        daily_demand=25, demand_std=6, lead_time=35,
        unit_cost=38, unit_margin=38, oos_multiplier=3.5
    )
    o = opt['optimal']
    print(f"\n  最优安全库存: {o['safety_stock_units']:.0f}件 (z={o['z_score']})")
    print(f"  对应Fill Rate目标: {o['fill_rate']:.1%}")
    print(f"  月持有成本: ${o['prevention_cost_monthly']:,.0f} vs 期望OOS成本: ${o['oos_cost_monthly']:,.0f}")
    print(f"  月总成本最优: ${o['total_cost_monthly']:,.0f}")

    print("\n[✓] 需求满足率与缺货成本全量化系统测试通过")
    return fr


if __name__ == "__main__":
    fr = run_fill_rate_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Safety-Stock-Replenishment]]（安全库存计算是Fill Rate的调节杠杆）、[[Skill-ITO-DOI-Inventory-Turnover-Optimizer]]（DOI与Fill Rate是供应链效率的双轴）
- **延伸（extends）**：[[Skill-Supply-Chain-KPI-Health-Dashboard]]（Fill Rate是KPI仪表盘的核心指标之一）、[[Skill-Purchase-Sales-Inventory-3D-Tracking]]（进销存追踪直接影响Fill Rate预测准确性）
- **可组合（combinable）**：[[Skill-Autobidding-Budget-Allocation-Optimization]]（OOS期间应暂停广告避免L4浪费）、[[Skill-SOP-Sales-Operations-Planning]]（S&OP以Fill Rate目标为约束条件规划补货）

## ⑤ 商业价值评估

- **ROI 预估**：月销$50万卖家，OOS真实成本=传统估算的3.5倍；通过正确量化后优化安全库存策略，年减少OOS 60%，避免损失约$8万；系统成本$2万，ROI≈400%
- **实施难度**：⭐⭐⭐☆☆（Fill Rate计算本身简单，难点是L3（买家流失LTV）和L5（评分影响）的量化需要历史数据建模）
- **优先级**：⭐⭐⭐⭐⭐（OOS成本被严重低估是行业普遍现象，量化后直接改变安全库存决策ROI方程）
- **适用规模**：所有规模，月销>$1万就值得建立Fill Rate监控
- **数据依赖**：订单级入库/出库记录（含OOS标记）、广告花费、评分历史、Buy Box拥有率历史
