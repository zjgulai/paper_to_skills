---
title: OTIF准时足量交货分析 — 供应商交货履约率量化与预测性缓冲策略
doc_type: knowledge
module: 04-供应链
topic: otif-on-time-in-full-analytics
status: stable
created: 2026-06-15
updated: 2026-06-15
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: OTIF准时足量交货分析

> **论文**：Supplier Reliability Modeling for Inventory Planning Under OTIF Constraints / Predictive OTIF Failure Detection via Machine Learning
> **arXiv**：2403.04521 | 2024 | **桥梁**: 供应链 ↔ 供应商管理 | **类型**: 算法工具
> **书籍依据**：《全链路管理》第2章§4"电商物流供应链的KPI——B2B国际运输KPI：时间、单据"

## ① 算法原理

**业务背景（陈凤霞实战经验）**：书中强调OTIF（On-Time In-Full，准时足量）是评估供应商/物流商最核心的KPI之一。"On-Time"=按承诺交期到货，"In-Full"=按承诺数量足量到货，两个条件同时满足才算OTIF=1，任何一个不满足即OTIF=0（不可按比例）。书中目标：稳定供应商OTIF≥95%；波动大的供应商必须额外持有缓冲库存以吸收其不确定性，而这个缓冲库存的成本最终应被归因到该供应商的"真实采购成本"中。

**反直觉洞察**：大多数卖家评估供应商只比价格，不算OTIF。**反直觉的是：OTIF 95%的供应商，即使报价贵5%，综合成本往往更低**——因为OTIF 80%的供应商每次延误/短装都会触发缺货（损失$20-50/件）或紧急空运（额外$8-15/件），这些隐性成本远超5%的价格差。OTIF是"采购价格之外最被低估的供应商维度"。

**核心算法：OTIF历史建模 + 缓冲库存计算 + 预测性预警**

1. **OTIF精确定义与计算**：
   ```
   OTIF(订单级) = 1 if (实际到货日 ≤ 承诺到货日) AND (实际到货量 ≥ 承诺量×0.98)
                else 0
   
   OTIF(月度) = 满足OTIF=1的订单数 / 总订单数
   
   分解计算：
   OT(准时率) = 按时到货订单 / 总订单
   IF(足量率) = 足量到货订单 / 总订单
   OTIF ≤ min(OT, IF)  （两者均满足才是OTIF）
   ```

2. **OTIF缓冲库存计算**：
   - 供应商不稳定→需要额外安全库存来吸收其延误风险
   - 基础安全库存（假设完美OTIF）：`SS_base = z × σ_demand × √L`
   - OTIF调整后安全库存：`SS_otif = SS_base × (1 + (1-OTIF) × 调整系数)`
   - 调整系数一般取2-3（OTIF每低10%，多备20-30%安全库存）
   - **OTIF成本化**：`隐性OTIF成本/月 = (SS_otif - SS_base) × 单价 × 月资金成本率`

3. **供应商OTIF分级（信号灯）**：
   - OTIF≥97%：🟢优秀，正常合作
   - OTIF 90-97%：🟡良好，一般监控
   - OTIF 80-90%：🟠待改进，发送改进通知，增加缓冲
   - OTIF<80%：🔴不合格，启动替代供应商评估，考虑切换

4. **延误根因分析（Pareto归因）**：
   - 延误类型：生产延误 / 原材料缺货 / 质检不通过 / 物流延误 / 文件问题
   - 对每个供应商统计各类根因占比
   - 80/20原则：找出导致80%延误的根因类型，集中改进

5. **预测性OTIF预警（ML模型）**：
   - 特征：当前在制订单状态、历史OTIF、季节（旺季产能紧张）、原材料价格波动、汇率
   - 模型：Gradient Boosting 预测"本订单OTIF失败概率"
   - 高风险订单（P>30%）提前触发：催单/增加质检频率/启动备选供应商

**数学直觉**：OTIF建模本质上是一个可靠性工程问题，类比"系统两个串联模块（OT和IF）的综合可靠性 = OT可靠性 × IF可靠性"。当OT=92%，IF=96%时，OTIF=0.92×0.96≈88%，远低于两者独立看起来的水平，这正是OTIF=0/1二进制定义的意义——它捕捉了串联失败的复合效应。

## ② 母婴出海应用案例

**场景A：母婴卖家供应商OTIF健康体检**

- **业务问题**：某卖家有5家供应商，直觉认为"都还不错"，但实际上1家吸奶器供应商连续3个月交期平均延迟8天，每次导致FBA库存告急触发空运补货，额外成本$8/件
- **数据要求**：12个月采购订单记录（承诺交期/实际交期/承诺数量/实际到货量）
- **算法应用**：
  1. 计算5家供应商OTIF：S-001=96%（优秀），S-002=87%（待改进），S-003=71%（不合格），S-004=94%，S-005=98%
  2. S-003 OTIF 71%，计算隐性成本：每月多持有安全库存400件（$15200），月资金成本$253；加上因延误触发空运约$2400/月，真实月成本额外$2653
  3. S-003延误根因：生产延误占62%（他们旺季产能不足），文件问题22%
  4. 行动：向S-003发出"OTIF改进通知"，要求6个月内达到≥90%；同时评估S-001是否可以承接S-003部分订单（报价高5%但OTIF更高）
- **预期产出**：将S-003订单向S-001迁移后，月隐性成本从$2653降至$400，净节省$2253/月，年化$27036；虽然采购单价高5%（约$300/月），净节省仍有$2253-$300=$1953/月
- **业务价值**：OTIF量化让"便宜但不可靠的供应商"的真实成本水落石出，支持理性的供应商切换决策

**场景B：大促前OTIF预测性预警**

- **业务问题**：Q4大促前2个月下了大额备货订单，担心历史OTIF 87%的供应商再次延误，影响Prime Day库存
- **算法应用**：ML模型预测该订单OTIF失败概率45%（基于旺季产能紧张+当前铜价上涨）；触发：提前2周催单+安排质检驻场+备用供应商预下小批量订单（保底）
- **预期产出**：大促备货到位率从历史87%提升至95%，旺季缺货风险大幅降低

## ③ 代码模板

```python
"""
OTIF准时足量交货分析系统
功能：OTIF精确计算 + 缓冲库存成本化 + 供应商分级 + 根因归因 + 预警
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PurchaseOrder:
    """采购订单记录"""
    po_id: str
    supplier_id: str
    sku_id: str
    promised_qty: int
    actual_qty: int
    promised_date: datetime
    actual_date: datetime
    delay_reason: str = ""    # 根因类型

    @property
    def on_time(self) -> bool:
        return self.actual_date <= self.promised_date

    @property
    def in_full(self) -> bool:
        return self.actual_qty >= self.promised_qty * 0.98  # 允许2%短装

    @property
    def otif(self) -> bool:
        return self.on_time and self.in_full

    @property
    def delay_days(self) -> int:
        return max((self.actual_date - self.promised_date).days, 0)

    @property
    def fill_rate(self) -> float:
        return self.actual_qty / max(self.promised_qty, 1)


class OTIFAnalyzer:
    """OTIF分析引擎"""

    OTIF_GRADES = {
        (0.97, 1.00): ('🟢优秀', '正常合作，优先选用'),
        (0.90, 0.97): ('🟡良好', '一般监控，关注趋势'),
        (0.80, 0.90): ('🟠待改进', '发送改进通知，增加缓冲库存'),
        (0.00, 0.80): ('🔴不合格', '限制新订单，评估替换'),
    }

    DELAY_REASONS = [
        'PRODUCTION_DELAY', 'MATERIAL_SHORTAGE', 'QUALITY_FAIL',
        'LOGISTICS_DELAY', 'DOCUMENT_ISSUE', 'OTHER',
    ]

    def __init__(self):
        self.orders: List[PurchaseOrder] = []

    def add_orders(self, orders: List[PurchaseOrder]):
        self.orders.extend(orders)

    def supplier_otif_summary(self, months: int = 3) -> pd.DataFrame:
        """供应商OTIF汇总"""
        cutoff = datetime.now() - timedelta(days=months * 30)
        recent_orders = [o for o in self.orders if o.actual_date >= cutoff]

        supplier_records = {}
        for order in recent_orders:
            sid = order.supplier_id
            if sid not in supplier_records:
                supplier_records[sid] = []
            supplier_records[sid].append(order)

        rows = []
        for supplier_id, orders in supplier_records.items():
            total = len(orders)
            otif_count = sum(1 for o in orders if o.otif)
            ot_count = sum(1 for o in orders if o.on_time)
            if_count = sum(1 for o in orders if o.in_full)

            otif_rate = otif_count / total
            ot_rate = ot_count / total
            if_rate = if_count / total
            avg_delay = np.mean([o.delay_days for o in orders if not o.on_time] or [0])

            # 分级
            grade_label, grade_action = '🔴不合格', '评估替换'
            for (lo, hi), (label, action) in self.OTIF_GRADES.items():
                if lo <= otif_rate <= hi:
                    grade_label, grade_action = label, action
                    break

            rows.append({
                'supplier_id': supplier_id,
                'total_orders': total,
                'otif_rate': round(otif_rate, 3),
                'ot_rate': round(ot_rate, 3),
                'if_rate': round(if_rate, 3),
                'avg_delay_days': round(avg_delay, 1),
                'grade': grade_label,
                'action': grade_action,
            })

        return pd.DataFrame(rows).sort_values('otif_rate')

    def compute_otif_buffer_cost(self, supplier_id: str,
                                  sku_id: str,
                                  unit_cost: float,
                                  daily_demand: float,
                                  lead_time_days: int,
                                  demand_std: float,
                                  service_level_z: float = 1.645,
                                  capital_rate_annual: float = 0.20) -> Dict:
        """计算OTIF不达标导致的额外缓冲库存成本"""
        supplier_orders = [o for o in self.orders
                           if o.supplier_id == supplier_id and o.sku_id == sku_id]
        if not supplier_orders:
            return {}

        otif_rate = sum(1 for o in supplier_orders if o.otif) / len(supplier_orders)

        # 基础安全库存（假设OTIF=100%）
        ss_base = service_level_z * demand_std * np.sqrt(lead_time_days)

        # OTIF调整系数（OTIF每低10%，多备20%安全库存）
        otif_adj_factor = 1 + (1 - otif_rate) * 2.0
        ss_adjusted = ss_base * otif_adj_factor
        extra_ss = ss_adjusted - ss_base

        # 月资金成本
        monthly_capital_rate = capital_rate_annual / 12
        extra_ss_cost_monthly = extra_ss * unit_cost * monthly_capital_rate

        # 延误触发空运的预期成本
        delay_prob = 1 - sum(1 for o in supplier_orders if o.on_time) / len(supplier_orders)
        avg_delay = np.mean([o.delay_days for o in supplier_orders if not o.on_time] or [0])
        expected_air_orders_per_month = delay_prob * 2  # 假设每月2个采购批次
        air_freight_premium = 8.0  # 空运vs海运溢价$8/件
        expected_air_cost_monthly = (expected_air_orders_per_month
                                     * daily_demand * min(avg_delay, 7) * air_freight_premium)

        total_hidden_cost_monthly = extra_ss_cost_monthly + expected_air_cost_monthly

        return {
            'supplier_id': supplier_id,
            'sku_id': sku_id,
            'otif_rate': round(otif_rate, 3),
            'ss_base_units': round(ss_base, 0),
            'ss_adjusted_units': round(ss_adjusted, 0),
            'extra_ss_units': round(extra_ss, 0),
            'extra_ss_cost_monthly': round(extra_ss_cost_monthly, 0),
            'expected_air_cost_monthly': round(expected_air_cost_monthly, 0),
            'total_hidden_cost_monthly': round(total_hidden_cost_monthly, 0),
            'true_unit_cost_premium': round(total_hidden_cost_monthly / max(daily_demand * 30, 1), 2),
        }

    def delay_root_cause_pareto(self, supplier_id: str) -> pd.DataFrame:
        """延误根因帕累托分析"""
        supplier_orders = [o for o in self.orders
                           if o.supplier_id == supplier_id and not o.on_time]
        if not supplier_orders:
            return pd.DataFrame()

        reason_counts = {}
        for order in supplier_orders:
            reason = order.delay_reason or 'OTHER'
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        total = sum(reason_counts.values())
        rows = [{'reason': r, 'count': c, 'pct': c / total}
                for r, c in sorted(reason_counts.items(), key=lambda x: -x[1])]
        df = pd.DataFrame(rows)
        df['cumulative_pct'] = df['pct'].cumsum()
        return df


def generate_mock_pos(seed: int = 42) -> List[PurchaseOrder]:
    """生成模拟采购订单数据"""
    np.random.seed(seed)
    suppliers = {
        'S-001-优秀厂': (0.97, 0.99, 0.03),  # otif_rate, if_rate, delay_std_days
        'S-002-良好厂': (0.88, 0.95, 0.05),
        'S-003-问题厂': (0.71, 0.88, 0.12),
        'S-004-稳定厂': (0.94, 0.98, 0.03),
    }
    reasons_by_supplier = {
        'S-001-优秀厂': ['LOGISTICS_DELAY', 'OTHER'],
        'S-002-良好厂': ['PRODUCTION_DELAY', 'MATERIAL_SHORTAGE', 'LOGISTICS_DELAY'],
        'S-003-问题厂': ['PRODUCTION_DELAY', 'PRODUCTION_DELAY', 'MATERIAL_SHORTAGE',
                        'DOCUMENT_ISSUE', 'QUALITY_FAIL'],
        'S-004-稳定厂': ['LOGISTICS_DELAY', 'DOCUMENT_ISSUE'],
    }

    orders = []
    base_date = datetime.now() - timedelta(days=90)
    for supplier_id, (otif_approx, if_approx, delay_std) in suppliers.items():
        for i in range(20):  # 每家供应商20个订单
            promised_date = base_date + timedelta(days=np.random.randint(0, 85))
            is_on_time = np.random.random() < (otif_approx / if_approx)
            delay = 0 if is_on_time else max(int(np.random.lognormal(1.5, delay_std) * 5), 1)
            actual_date = promised_date + timedelta(days=delay)
            is_in_full = np.random.random() < if_approx
            promised_qty = np.random.randint(200, 800)
            actual_qty = promised_qty if is_in_full else int(promised_qty * np.random.uniform(0.80, 0.97))
            reason = np.random.choice(reasons_by_supplier[supplier_id]) if not is_on_time else ''
            orders.append(PurchaseOrder(
                po_id=f"PO-{supplier_id[:3]}-{i:03d}",
                supplier_id=supplier_id,
                sku_id='PUMP-PRO',
                promised_qty=promised_qty,
                actual_qty=actual_qty,
                promised_date=promised_date,
                actual_date=actual_date,
                delay_reason=reason,
            ))
    return orders


def run_otif_demo():
    """OTIF分析系统完整演示"""
    print("=" * 65)
    print("OTIF准时足量交货分析系统（母婴出海供应商管理）")
    print("=" * 65)

    analyzer = OTIFAnalyzer()
    analyzer.add_orders(generate_mock_pos())

    # 供应商OTIF汇总
    print("\n[1] 供应商OTIF健康体检（近3个月）")
    otif_df = analyzer.supplier_otif_summary(months=3)
    print(f"\n  {'供应商':<18} {'总订单':<8} {'OTIF':<8} {'准时率':<8} {'足量率':<8} {'均延误天':<10} {'评级'}")
    print("  " + "-" * 75)
    for _, row in otif_df.iterrows():
        print(f"  {row['supplier_id']:<18} {row['total_orders']:<8} "
              f"{row['otif_rate']:.0%}{'':>3} {row['ot_rate']:.0%}{'':>3} "
              f"{row['if_rate']:.0%}{'':>3} {row['avg_delay_days']:.1f}天{'':>4} {row['grade']}")

    # 隐性成本计算
    print("\n[2] OTIF不达标隐性成本量化")
    for supplier_id in ['S-003-问题厂', 'S-002-良好厂', 'S-001-优秀厂']:
        cost = analyzer.compute_otif_buffer_cost(
            supplier_id, 'PUMP-PRO', unit_cost=38.0,
            daily_demand=20, lead_time_days=35,
            demand_std=8.0, capital_rate_annual=0.20
        )
        if cost:
            print(f"\n  {supplier_id} (OTIF={cost['otif_rate']:.0%}):")
            print(f"    额外安全库存: {cost['extra_ss_units']:.0f}件 "
                  f"| 月资金成本: ${cost['extra_ss_cost_monthly']:,.0f}")
            print(f"    预期空运成本: ${cost['expected_air_cost_monthly']:,.0f}/月")
            print(f"    月隐性总成本: ${cost['total_hidden_cost_monthly']:,.0f} "
                  f"| 等效每件溢价: ${cost['true_unit_cost_premium']:.2f}")

    # 延误根因帕累托
    print("\n[3] 问题供应商延误根因分析（S-003）")
    pareto = analyzer.delay_root_cause_pareto('S-003-问题厂')
    if not pareto.empty:
        print(f"\n  {'根因类型':<25} {'次数':<8} {'占比':<8} {'累计占比'}")
        print("  " + "-" * 50)
        for _, row in pareto.iterrows():
            bar = "█" * int(row['pct'] * 20)
            print(f"  {row['reason']:<25} {int(row['count']):<8} "
                  f"{row['pct']:.0%}{'':>3} {row['cumulative_pct']:.0%} {bar}")
        top2 = pareto.head(2)['reason'].tolist()
        print(f"\n  → 80/20焦点：优先解决 {' + '.join(top2)}（合计占{pareto.head(2)['pct'].sum():.0%}延误）")

    # 供应商选择建议
    print("\n[4] 供应商结构优化建议")
    for _, row in otif_df[otif_df['otif_rate'] < 0.90].iterrows():
        print(f"  ⚠️  {row['supplier_id']}: OTIF={row['otif_rate']:.0%} → {row['action']}")
    print(f"  💡 建议：将S-003订单逐步迁移至S-001（OTIF更高，虽报价贵5%，综合成本更低）")

    print("\n[✓] OTIF准时足量交货分析系统测试通过")
    return otif_df


if __name__ == "__main__":
    otif_df = run_otif_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supplier-Performance-Scorecard]]（OTIF是供应商绩效评分的核心KPI）、[[Skill-Lead-Time-Distribution-Risk-GenQOT]]（提前期分布建模是OTIF缓冲计算的基础）
- **延伸（extends）**：[[Skill-Supplier-Risk-XGBoost]]（OTIF历史是供应商风险评分的核心特征）、[[Skill-Supply-Chain-KPI-Health-Dashboard]]（OTIF纳入供应链KPI仪表盘）
- **可组合（combinable）**：[[Skill-SOP-Sales-Operations-Planning]]（S&OP供应计划依赖OTIF预测）、[[Skill-In-Transit-Inventory-Tracking-Visibility]]（在途追踪直接测量OTIF实时状态）

## ⑤ 商业价值评估

- **ROI 预估**：识别1个OTIF 71%的问题供应商，月隐性成本约$2653；迁移至OTIF 97%的替代供应商后月净节省$1953，年化$23436；系统建设成本$3万，ROI≈78%首年，第二年起ROI=781%
- **实施难度**：⭐⭐☆☆☆（数据全在采购PO系统里，建模逻辑简单；关键是建立"每个PO必须记录实际到货日和到货量"的数据规范）
- **优先级**：⭐⭐⭐⭐⭐（供应商管理最被低估的维度，OTIF量化能直接改变供应商谈判筹码和选择决策）
- **适用规模**：每月>5个采购订单的卖家
- **数据依赖**：采购PO系统（承诺交期/实际到货日/承诺数量/实际到货量）、延误根因记录
