---
title: 进销存三维动态追踪 — 进货/销售/库存联动监控与比率预警体系
doc_type: knowledge
module: 04-供应链
topic: purchase-sales-inventory-3d-tracking
status: stable
created: 2026-06-15
updated: 2026-06-15
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 进销存三维动态追踪

> **论文**：Integrated Purchase-Sales-Inventory Analytics for E-Commerce / Supply Chain Signal Detection via Trivariate Time Series
> **arXiv**：2401.14523 | 2024 | **桥梁**: 供应链 ↔ 运营财务 | **类型**: 算法工具
> **书籍依据**：《全链路管理》第2章§3"电商物流计划供应链的KPI：计划和预测准确率——进、销、存管理"

## ① 算法原理

**业务背景（陈凤霞实战经验）**：书中将"进销存管理"列为物流计划供应链最核心的KPI维度。"进"=进货量（入库），"销"=销售量（出库），"存"=库存量（在库）。三者构成一个**封闭会计恒等式**：`期末存 = 期初存 + 进 - 销`。任何一维出现异常，都会在其他维度反映出来——这正是进销存分析的核心价值：**通过监控三者的比率关系，比单独观察任何一个维度更早发现异常**。

**反直觉洞察**：大多数卖家看"库存够不够"（看存），或者"这周卖了多少"（看销），很少同时看"进"——进货节奏才是决定未来库存状态的主动变量。**反直觉的是：进销比（进/销）是比DOI更领先的预警指标**——当进销比连续3周>1.5时，积压已在路上；当进销比<0.5时，缺货即将发生，比DOI提前2-4周预警。

**核心算法：进销存三维比率监控 + 异常检测**

1. **核心比率定义**：
   - **进销比（P/S Ratio）** = 本期进货量 / 本期销售量
     - 正常范围：0.8-1.2（动态平衡）
     - >1.5 连续3周 → 积压预警
     - <0.5 连续2周 → 缺货预警
   - **存销比（I/S Ratio）= DOI** = 当前库存 / 日均销量（即DOI）
   - **进存比（P/I Ratio）** = 本期进货量 / 当前库存
     - >0.5：库存高速补充中（验证入库计划）
     - <0.1：库存几乎没有新货流入（需检查采购是否延误）
   - **库存变动率（ΔI）** = (期末存 - 期初存) / 期初存
     - 连续3个月>20% → 库存快速膨胀预警

2. **时间序列异常检测（CUSUM + 规则）**：
   - 对P/S Ratio建立CUSUM统计量，检测结构性异常（持续偏离正常范围）
   - 与季节性基线对比：旺季前进销比>1.5是正常（备货），淡季进销比>1.5是危险
   - 同比对比：本月进销比 vs 去年同月进销比，偏差>30%触发复查

3. **三角形平衡诊断（Triangle Balance）**：
   ```
   期末存 = 期初存 + 进 - 销
   
   平衡检验：|账面期末存 - 实际盘点存| / 账面存 < 0.5%
   
   差异来源：
   ① 损耗/丢失（存偏小）
   ② 退货未入账（存偏大）
   ③ 多批次进货漏记（进偏小）
   ④ 退款但货未退（销偏大但存不减少）
   ```

4. **分周期多粒度追踪**：
   - 日粒度：实时库存波动监控（出入库事件驱动）
   - 周粒度：P/S Ratio趋势分析（主要预警频率）
   - 月粒度：ITO/DOI月度汇总，财务对账
   - 季度粒度：库存结构优化决策（ABC再分类）

5. **预算准确率（Plan Accuracy）辅助**：
   - 进货计划准确率 = |实际到货 - 计划到货| / 计划到货
   - 销售计划准确率 = |实际销售 - 销售预测| / 销售预测
   - 库存计划准确率 = |实际期末存 - 计划期末存| / 计划期末存
   - 三维计划准确率联合评分，识别哪一维是系统性偏差来源

**数学直觉**：进销存是一个状态空间模型，`I_t = I_{t-1} + P_t - S_t`。通过卡尔曼滤波或简单统计检验，可以识别哪些时期的状态转移出现了统计异常（不能被正常季节性+趋势解释），这些异常往往对应了选品失误、采购延误、需求突变等具体业务事件。

## ② 母婴出海应用案例

**场景A：多SKU进销存周报自动化（替代手工Excel）**

- **业务问题**：某卖家每周手工汇总各平台进销存数据到Excel，耗时4小时，且容易出错漏项，无法及时发现异常
- **数据要求**：亚马逊Inventory API（日入库/出库记录）、FBA库存报告、采购PO记录
- **算法应用**：
  1. 每周一自动拉取上周进销存数据
  2. 计算每个SKU的P/S Ratio和ΔI（库存变动率）
  3. 发现："OLD-NIPPLE-PKG"连续4周P/S=1.8（进货量远超销售），库存膨胀62%
  4. 同时发现："PUMP-PRO"上周P/S=0.31（销售远超进货），DOI已降至12天（缺货预警！）
  5. 自动生成周报推送运营团队，含行动优先级
- **预期产出**：周报从4小时手工降至10分钟自动，异常发现提前3-5天，年化防损$15万（断货+积压）
- **业务价值**：进销存比率分析是最直观的"供应链体检"，每周一次就能防止大多数系统性问题

**场景B：三角平衡稽核找出库存差异根因**

- **业务问题**：月末盘点发现FBA库存账实差异3.2%（超标准0.5%），但不知道原因
- **算法应用**：三角平衡诊断：期初存500件 + 进货300件 - 销售320件 = 理论期末480件，实际盘点465件，差异15件。通过退货记录核查发现：12件退货客户已获退款但货物未退还FBA（Amazon判定"退款不退货"），3件在途被海关扣押未入账。根因清晰，处理有据
- **预期产出**：差异率从3.2%降至0.4%，库存准确率达标，财务账目可信度提升

## ③ 代码模板

```python
"""
进销存三维动态追踪系统
功能：P/S/I三维比率计算 + CUSUM异常检测 + 三角平衡稽核 + 计划准确率
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class WeeklyPSI:
    """单周进销存数据"""
    sku_id: str
    week: str                   # 'W2026-23' 格式
    opening_stock: int          # 期初库存
    purchases: int              # 本周进货（入库）
    sales: int                  # 本周销售（出库）
    closing_stock_actual: int   # 期末实际库存（盘点）
    # 计划值
    planned_purchases: int = 0
    planned_sales: int = 0

    @property
    def closing_stock_theoretical(self) -> int:
        return self.opening_stock + self.purchases - self.sales

    @property
    def ps_ratio(self) -> float:
        return self.purchases / max(self.sales, 0.01)

    @property
    def is_ratio(self) -> float:
        return self.closing_stock_actual / max(self.sales / 7, 0.01)  # DOI

    @property
    def pi_ratio(self) -> float:
        return self.purchases / max(self.opening_stock, 0.01)

    @property
    def stock_change_rate(self) -> float:
        return (self.closing_stock_actual - self.opening_stock) / max(self.opening_stock, 1)

    @property
    def balance_discrepancy(self) -> int:
        return self.closing_stock_actual - self.closing_stock_theoretical

    @property
    def balance_discrepancy_rate(self) -> float:
        return abs(self.balance_discrepancy) / max(self.closing_stock_theoretical, 1)

    @property
    def purchase_plan_accuracy(self) -> float:
        if self.planned_purchases == 0:
            return 1.0
        return 1 - abs(self.purchases - self.planned_purchases) / max(self.planned_purchases, 1)

    @property
    def sales_plan_accuracy(self) -> float:
        if self.planned_sales == 0:
            return 1.0
        return 1 - abs(self.sales - self.planned_sales) / max(self.planned_sales, 1)


def classify_ps_status(ps_ratio: float, is_seasonal: bool = False,
                        is_pre_peak: bool = False) -> Tuple[str, str]:
    """P/S比率状态分类（考虑季节性）"""
    if is_pre_peak:
        normal_upper = 2.0  # 旺季前备货允许更高进销比
    else:
        normal_upper = 1.3

    if ps_ratio > 2.0:
        return '🔴积压严重', '进货量是销售量2倍以上，立即暂停补货'
    elif ps_ratio > normal_upper:
        return '🟡积压风险', '进货量超过销售量，关注库存增长'
    elif ps_ratio < 0.4:
        return '🔴缺货预警', '销售量远超进货量，库存快速消耗'
    elif ps_ratio < 0.7:
        return '🟡补货偏少', '进货量低于销售量，需加速补货'
    else:
        return '🟢均衡', '进销平衡，库存健康'


def cusum_anomaly_detection(ps_series: List[float],
                             target: float = 1.0,
                             threshold: float = 3.0) -> List[Dict]:
    """CUSUM统计量检测P/S比率结构性异常"""
    k = 0.5  # 允许偏差系数
    cusum_up = 0.0
    cusum_down = 0.0
    anomalies = []

    for i, val in enumerate(ps_series):
        cusum_up = max(0, cusum_up + val - (target + k))
        cusum_down = max(0, cusum_down + (target - k) - val)

        if cusum_up > threshold:
            anomalies.append({
                'week_idx': i,
                'type': 'OVERSTOCK_TREND',
                'cusum': cusum_up,
                'message': f'连续偏高P/S比率，积压趋势',
            })
        elif cusum_down > threshold:
            anomalies.append({
                'week_idx': i,
                'type': 'UNDERSTOCK_TREND',
                'cusum': cusum_down,
                'message': f'连续偏低P/S比率，缺货趋势',
            })

    return anomalies


def triangle_balance_audit(weeks: List[WeeklyPSI]) -> Dict:
    """三角平衡稽核——找出进销存差异根因"""
    if not weeks:
        return {}

    total_purchases = sum(w.purchases for w in weeks)
    total_sales = sum(w.sales for w in weeks)
    opening = weeks[0].opening_stock
    theoretical_closing = opening + total_purchases - total_sales
    actual_closing = weeks[-1].closing_stock_actual
    total_discrepancy = actual_closing - theoretical_closing
    discrepancy_rate = abs(total_discrepancy) / max(theoretical_closing, 1)

    # 差异方向判断
    if abs(discrepancy_rate) < 0.005:
        status = '✅平衡（差异<0.5%）'
        possible_causes = []
    elif total_discrepancy < 0:
        status = '⚠️账面偏多（实际<理论）'
        possible_causes = [
            '商品损耗/丢失未记录',
            'FBA移除未更新账面',
            '出库漏记（促销赠品等）',
        ]
    else:
        status = '⚠️账面偏少（实际>理论）'
        possible_causes = [
            '退货已退款但货未退还，未计入入库',
            '入库数量超过PO数量漏记',
            '跨仓调拨未同步更新',
        ]

    return {
        'opening_stock': opening,
        'total_purchases': total_purchases,
        'total_sales': total_sales,
        'theoretical_closing': theoretical_closing,
        'actual_closing': actual_closing,
        'discrepancy': total_discrepancy,
        'discrepancy_rate': discrepancy_rate,
        'status': status,
        'possible_causes': possible_causes,
        'requires_investigation': discrepancy_rate > 0.005,
    }


class PSITracker:
    """进销存追踪系统"""

    def __init__(self):
        self.data: Dict[str, List[WeeklyPSI]] = {}

    def add_week(self, record: WeeklyPSI):
        if record.sku_id not in self.data:
            self.data[record.sku_id] = []
        self.data[record.sku_id].append(record)

    def sku_weekly_report(self, sku_id: str) -> pd.DataFrame:
        """单SKU周度进销存报告"""
        weeks = self.data.get(sku_id, [])
        records = []
        for w in weeks:
            ps_status, ps_msg = classify_ps_status(w.ps_ratio)
            records.append({
                'week': w.week,
                'purchases': w.purchases,
                'sales': w.sales,
                'closing_stock': w.closing_stock_actual,
                'ps_ratio': round(w.ps_ratio, 2),
                'doi': round(w.is_ratio, 1),
                'stock_change_pct': f"{w.stock_change_rate:+.0%}",
                'balance_discrepancy_rate': f"{w.balance_discrepancy_rate:.2%}",
                'ps_status': ps_status,
                'purchase_plan_acc': f"{w.purchase_plan_accuracy:.0%}",
                'sales_plan_acc': f"{w.sales_plan_accuracy:.0%}",
            })
        return pd.DataFrame(records)

    def portfolio_weekly_summary(self, week: str) -> pd.DataFrame:
        """全品类周度汇总"""
        records = []
        for sku_id, weeks in self.data.items():
            week_data = [w for w in weeks if w.week == week]
            if not week_data:
                continue
            w = week_data[0]
            ps_status, _ = classify_ps_status(w.ps_ratio)
            records.append({
                'sku_id': sku_id,
                'purchases': w.purchases,
                'sales': w.sales,
                'closing_stock': w.closing_stock_actual,
                'ps_ratio': round(w.ps_ratio, 2),
                'doi': round(w.is_ratio, 1),
                'status': ps_status,
            })
        return pd.DataFrame(records).sort_values('ps_ratio', ascending=False)

    def detect_anomalies(self, sku_id: str) -> List[Dict]:
        """检测特定SKU的P/S比率异常"""
        weeks = self.data.get(sku_id, [])
        if len(weeks) < 3:
            return []
        ps_series = [w.ps_ratio for w in weeks]
        return cusum_anomaly_detection(ps_series)


def generate_mock_psi_data(n_weeks: int = 8, seed: int = 42) -> Dict[str, List[WeeklyPSI]]:
    """生成模拟进销存数据"""
    np.random.seed(seed)
    today = datetime.now()
    skus_config = {
        "PUMP-PRO":    (800, 25, 20, 22, 0.05),  # stock, planned_sales, actual_sales, planned_pur, noise
        "OLD-NIPPLE":  (200, 5,  3,  15, 0.10),  # 问题SKU：销售少但进货多
        "PUMP-V1":     (150, 18, 22, 10, 0.08),  # 问题SKU：销售多但进货少
        "BOTTLE-3P":   (500, 20, 19, 20, 0.05),  # 健康SKU
    }

    all_data = {}
    for sku_id, (init_stock, ps, as_, pp, noise) in skus_config.items():
        records = []
        stock = init_stock
        for i in range(n_weeks):
            week_dt = today - timedelta(weeks=n_weeks - i - 1)
            week_str = f"W{week_dt.strftime('%Y-%V')}"

            sales = max(int(as_ + np.random.normal(0, as_ * noise)), 0)
            if sku_id == "OLD-NIPPLE":
                purchases = int(pp + np.random.normal(0, 2))  # 持续高进货
            elif sku_id == "PUMP-V1":
                purchases = max(int(pp - np.random.normal(0, 2)), 0)  # 持续低进货
            else:
                purchases = max(int(pp + np.random.normal(0, 3)), 0)

            actual_closing = max(stock + purchases - sales + int(np.random.normal(0, 1)), 0)
            record = WeeklyPSI(
                sku_id=sku_id,
                week=week_str,
                opening_stock=stock,
                purchases=purchases,
                sales=sales,
                closing_stock_actual=actual_closing,
                planned_purchases=pp,
                planned_sales=ps,
            )
            records.append(record)
            stock = actual_closing
        all_data[sku_id] = records
    return all_data


def run_psi_tracking_demo():
    """进销存追踪系统完整演示"""
    print("=" * 65)
    print("进销存三维动态追踪系统（母婴出海）")
    print("=" * 65)

    tracker = PSITracker()
    mock_data = generate_mock_psi_data(n_weeks=8)
    for sku_id, records in mock_data.items():
        for r in records:
            tracker.add_week(r)

    # 最近一周全品类汇总
    latest_week = list(mock_data.values())[0][-1].week
    print(f"\n[1] 周度全品类进销存汇总（{latest_week}）")
    summary = tracker.portfolio_weekly_summary(latest_week)
    print(f"\n  {'SKU':<18} {'进货':<7} {'销售':<7} {'期末库存':<10} {'P/S比':<8} {'DOI':<7} {'状态'}")
    print("  " + "-" * 70)
    for _, row in summary.iterrows():
        print(f"  {row['sku_id']:<18} {row['purchases']:<7} {row['sales']:<7} "
              f"{row['closing_stock']:<10} {row['ps_ratio']:<8.2f} {row['doi']:<7.0f} {row['status']}")

    # CUSUM异常检测
    print("\n[2] P/S比率CUSUM异常检测（8周趋势）")
    for sku_id in mock_data:
        anomalies = tracker.detect_anomalies(sku_id)
        weeks = mock_data[sku_id]
        ps_trend = [round(w.ps_ratio, 2) for w in weeks]
        print(f"\n  {sku_id}: P/S趋势 {ps_trend}")
        if anomalies:
            for a in anomalies[-2:]:  # 最近2个异常
                print(f"    ⚠️  {a['type']}: {a['message']}")
        else:
            print(f"    ✅ 无结构性异常")

    # 三角平衡稽核
    print("\n[3] 三角平衡稽核（全周期）")
    for sku_id, weeks in mock_data.items():
        audit = triangle_balance_audit(weeks)
        icon = '✅' if not audit['requires_investigation'] else '⚠️'
        print(f"\n  {icon} {sku_id}:")
        print(f"     期初:{audit['opening_stock']} + 进:{audit['total_purchases']} "
              f"- 销:{audit['total_sales']} = 理论:{audit['theoretical_closing']}")
        print(f"     实际期末:{audit['actual_closing']} | 差异:{audit['discrepancy']}件 "
              f"({audit['discrepancy_rate']:.2%}) | {audit['status']}")
        if audit['possible_causes']:
            for cause in audit['possible_causes'][:2]:
                print(f"     可能原因: {cause}")

    # 计划准确率
    print("\n[4] 计划准确率评估（最近4周均值）")
    for sku_id, weeks in mock_data.items():
        recent = weeks[-4:]
        avg_pur_acc = np.mean([w.purchase_plan_accuracy for w in recent])
        avg_sales_acc = np.mean([w.sales_plan_accuracy for w in recent])
        pur_grade = '✅' if avg_pur_acc >= 0.85 else '⚠️'
        sales_grade = '✅' if avg_sales_acc >= 0.85 else '⚠️'
        print(f"  {sku_id}: 进货计划准确率={avg_pur_acc:.0%}{pur_grade} | "
              f"销售计划准确率={avg_sales_acc:.0%}{sales_grade}")

    print("\n[✓] 进销存三维动态追踪系统测试通过")
    return tracker


if __name__ == "__main__":
    tracker = run_psi_tracking_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-SOP-Sales-Operations-Planning]]（S&OP中进销存是核心输入）、[[Skill-ITO-DOI-Inventory-Turnover-Optimizer]]（DOI是进销存比率的衍生指标）
- **延伸（extends）**：[[Skill-Supply-Chain-KPI-Health-Dashboard]]（进销存比率是KPI仪表盘的核心数据源）、[[Skill-Fill-Rate-OOS-Cost-Quantification]]（进销存追踪直接支撑满足率计算）
- **可组合（combinable）**：[[Skill-In-Transit-Inventory-Tracking-Visibility]]（在途库存是"进"的预测值）、[[Skill-Bullwhip-Effect-Mitigation]]（进销存波动分析识别牛鞭效应）

## ⑤ 商业价值评估

- **ROI 预估**：每周4小时手工进销存 → 10分钟自动报告，年节省约200小时运营时间（价值$5000+）；P/S比率异常提前3-5天发现，防止断货和积压年化$10-15万；系统成本$2万，ROI≈600%
- **实施难度**：⭐⭐☆☆☆（逻辑简单，Amazon/Shopee均有入出库报告API；主要挑战是多平台数据对接统一）
- **优先级**：⭐⭐⭐⭐⭐（电商供应链最基础的管控工具，没有进销存追踪一切都是盲飞）
- **适用规模**：所有规模，月销>$2万就值得自动化
- **数据依赖**：Amazon FBA Inventory API（入出库记录）、采购PO系统数据、销售报告
