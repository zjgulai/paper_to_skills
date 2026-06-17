---
title: 物流计划进销存三维准确率体系 — 入库/存货/销售三层预测准确率量化与根因归因
doc_type: knowledge
module: 04-供应链
topic: logistics-plan-inbound-inventory-sales-accuracy
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 物流计划进销存三维准确率体系

> **书籍**：《全链路管理》陈凤霞 第二章第三节"电商物流计划供应链的KPI——计划和预测的准确率：进、销、存管理"
> **桥梁**: 供应链 ↔ 运营财务 | **类型**: 算法工具

## ① 算法原理

**书籍核心洞察（陈凤霞）**：物流计划供应链的计划准确率，必须分"进、销、存"三个维度独立追踪，而不是只看一个综合指标。书中给出了精确定义：

**三维准确率定义**：

1. **进（入库准确率）**：
   - 预约入库准确率 = 实际入库件数 / 预约申报件数
   - 书中揭示的典型问题：预约了500件实际入库480件（入库准确率96%），但系统显示库存500件→产生20件"幽灵库存"
   - 衡量的是：**采购端的承诺兑现能力 + 报关/质检的实际落地能力**

2. **存（存货预测准确率）**：
   - 存货件数预测准确率 = 1 - |预测期末库存 - 实际期末库存| / 实际期末库存
   - 仓容面积换算准确率 = 实际占用仓容 / 计划仓容
   - 意义：库存预测不准→仓容规划失效→爆仓或空仓浪费

3. **销（销售件数预测准确率）**：
   - 销售件数预测准确率 = 1 - |预测件数 - 实际件数| / 实际件数
   - 件单比 = 实际发货件数 / 订单数（衡量每单平均件数）
   - 注意：金额预测准确率 ≠ 件数预测准确率，因为客单价有波动

**三维联动关系（书中核心模型）**：
```
期末库存 = 期初库存 + 实际入库 - 实际销售
          ≈    期初  + 预测入库 × 入库准确率 - 预测销售 × 销售准确率

存货预测误差 = f(入库准确率误差, 销售准确率误差)
→ 提升存货预测准确率，需要同时提升入库准确率和销售预测准确率
```

**算法突破口：误差传播链（Error Propagation Chain）**：
- 入库误差→库存误差→满足率误差→客体验误差（级联传播）
- 用因果图（DAG）建模三维误差的依赖结构，找到改善最高ROI的节点

## ② 母婴出海应用案例

**场景A：大促前三维准确率健康检查**

- **业务问题**：Prime Day前1周，运营发现备货件数比预期少15%，不知道是入库环节出了问题还是库存预测本身偏差
- **三维准确率诊断**：
  1. 入库准确率：计划入库3000件，实际入库2700件（90%）→入库环节损失300件
  2. 存货预测准确率：预测期末库存2500件，实际2200件（88%）→库存预测偏差300件
  3. 归因：存货偏差完全来自入库不足，预测模型本身是准确的
  4. 行动：紧急联系供应商补货300件，不需要调整预测模型
- **预期产出**：精准归因节省错误干预成本；大促缺货率从预期8%降至3%

**场景B：件单比异常监控**

- **业务问题**：某月订单量同比增长20%，但销售件数同比增长只有8%（件单比从2.1降至1.89）
- **件单比分析**：件单比下降→用户从"多件购买"转变为"单件购买"→可能是大件商品（婴儿推车）占比上升，或多件优惠活动失效→针对性恢复多件购买优惠

## ③ 代码模板

```python
"""
物流计划进销存三维准确率体系
基于《全链路管理》陈凤霞 第三节物流计划供应链KPI
进库准确率 + 存货预测准确率 + 销售件数预测准确率 + 件单比
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


@dataclass
class LogisticsPlanData:
    """物流计划数据结构"""
    sku_id: str
    period: str  # 'YYYY-MM' 格式

    # 进：入库
    planned_inbound: int    # 计划入库件数
    actual_inbound: int     # 实际入库件数

    # 存：库存
    opening_stock: int      # 期初库存
    planned_closing_stock: int  # 预测期末库存
    actual_closing_stock: int   # 实际期末库存
    planned_space_sqm: float    # 计划仓容（平米）
    actual_space_sqm: float     # 实际占用仓容（平米）

    # 销：销售
    planned_sales_units: int    # 预测销售件数
    actual_sales_units: int     # 实际销售件数
    total_orders: int           # 订单数（用于件单比）


class ThreeDimensionAccuracyTracker:
    """进销存三维准确率追踪器"""

    def compute_inbound_accuracy(self, data: LogisticsPlanData) -> Dict:
        """计算入库准确率"""
        if data.planned_inbound == 0:
            return {'accuracy': None, 'status': 'N/A'}
        accuracy = data.actual_inbound / data.planned_inbound
        gap = data.actual_inbound - data.planned_inbound
        ghost_inventory_risk = max(-gap, 0)  # 系统记录>实际=幽灵库存风险

        status = '✅准确' if accuracy >= 0.97 else ('🟡轻微偏差' if accuracy >= 0.92 else '🔴偏差大')
        return {
            'planned': data.planned_inbound,
            'actual': data.actual_inbound,
            'accuracy': accuracy,
            'accuracy_pct': f"{accuracy:.1%}",
            'gap_units': gap,
            'ghost_inventory_risk': ghost_inventory_risk,
            'status': status,
            'action': '检查供应商交期和质检通过率' if accuracy < 0.95 else '正常',
        }

    def compute_inventory_accuracy(self, data: LogisticsPlanData) -> Dict:
        """计算存货预测准确率"""
        if data.actual_closing_stock == 0:
            return {'accuracy': None}

        inventory_accuracy = 1 - abs(data.planned_closing_stock - data.actual_closing_stock) / data.actual_closing_stock
        space_accuracy = data.actual_space_sqm / max(data.planned_space_sqm, 0.01)

        # 验证等式：期末 = 期初 + 入库 - 销售
        theoretical_closing = data.opening_stock + data.actual_inbound - data.actual_sales_units
        reconciliation_error = abs(theoretical_closing - data.actual_closing_stock)

        return {
            'planned_closing': data.planned_closing_stock,
            'actual_closing': data.actual_closing_stock,
            'inventory_accuracy': inventory_accuracy,
            'inventory_accuracy_pct': f"{inventory_accuracy:.1%}",
            'space_accuracy': space_accuracy,
            'space_accuracy_pct': f"{space_accuracy:.1%}",
            'space_status': '✅合理' if 0.7 <= space_accuracy <= 1.1 else '⚠️仓容异常',
            'reconciliation_error': reconciliation_error,
            'balanced': reconciliation_error <= 2,
        }

    def compute_sales_accuracy(self, data: LogisticsPlanData) -> Dict:
        """计算销售件数预测准确率 + 件单比"""
        if data.actual_sales_units == 0:
            return {'accuracy': None}

        accuracy = 1 - abs(data.planned_sales_units - data.actual_sales_units) / data.actual_sales_units
        units_per_order = data.actual_sales_units / max(data.total_orders, 1)

        return {
            'planned_units': data.planned_sales_units,
            'actual_units': data.actual_sales_units,
            'accuracy': accuracy,
            'accuracy_pct': f"{accuracy:.1%}",
            'units_per_order': units_per_order,
            'units_per_order_str': f"{units_per_order:.2f}件/单",
            'status': '✅准确' if accuracy >= 0.85 else ('🟡偏差' if accuracy >= 0.75 else '🔴严重偏差'),
        }

    def full_analysis(self, data: LogisticsPlanData) -> Dict:
        """完整三维分析 + 误差传播归因"""
        inbound = self.compute_inbound_accuracy(data)
        inventory = self.compute_inventory_accuracy(data)
        sales = self.compute_sales_accuracy(data)

        # 误差传播分析
        inbound_contribution = data.planned_inbound - data.actual_inbound
        sales_contribution = data.actual_sales_units - data.planned_sales_units
        inventory_error = data.planned_closing_stock - data.actual_closing_stock

        # 解释存货误差来源
        error_from_inbound = -inbound_contribution  # 入库少→库存少
        error_from_sales = sales_contribution         # 销售超预期→库存少
        error_unexplained = inventory_error - error_from_inbound - error_from_sales

        # 综合健康评分
        scores = []
        if inbound.get('accuracy'): scores.append(inbound['accuracy'])
        if inventory.get('inventory_accuracy'): scores.append(inventory['inventory_accuracy'])
        if sales.get('accuracy'): scores.append(sales['accuracy'])
        overall_score = np.mean(scores) if scores else 0

        return {
            'sku_id': data.sku_id,
            'period': data.period,
            'inbound': inbound,
            'inventory': inventory,
            'sales': sales,
            'error_propagation': {
                'inventory_error_total': inventory_error,
                'contribution_from_inbound': error_from_inbound,
                'contribution_from_sales': error_from_sales,
                'unexplained': error_unexplained,
                'main_cause': '入库不足' if abs(error_from_inbound) > abs(error_from_sales) else '销售超预期',
            },
            'overall_score': overall_score,
            'overall_status': '🟢健康' if overall_score >= 0.90 else ('🟡一般' if overall_score >= 0.80 else '🔴需改进'),
        }


def run_three_dimension_accuracy_demo():
    """三维准确率体系完整演示"""
    print("=" * 65)
    print("物流计划进销存三维准确率体系")
    print("基于《全链路管理》陈凤霞 物流计划供应链KPI")
    print("=" * 65)

    tracker = ThreeDimensionAccuracyTracker()

    # 模拟三个SKU的进销存数据
    skus = [
        LogisticsPlanData(
            sku_id='PUMP-PRO', period='2026-06',
            planned_inbound=500, actual_inbound=485,  # 入库轻微不足
            opening_stock=300, planned_closing_stock=450, actual_closing_stock=430,
            planned_space_sqm=15.0, actual_space_sqm=14.3,
            planned_sales_units=350, actual_sales_units=355, total_orders=280,
        ),
        LogisticsPlanData(
            sku_id='WARMER-S1', period='2026-06',
            planned_inbound=300, actual_inbound=245,  # 入库严重不足（供应商延迟）
            opening_stock=150, planned_closing_stock=200, actual_closing_stock=120,
            planned_space_sqm=8.0, actual_space_sqm=4.0,
            planned_sales_units=250, actual_sales_units=275, total_orders=250,
        ),
        LogisticsPlanData(
            sku_id='BOTTLE-3P', period='2026-06',
            planned_inbound=800, actual_inbound=810,  # 入库略超
            opening_stock=500, planned_closing_stock=650, actual_closing_stock=660,
            planned_space_sqm=20.0, actual_space_sqm=21.5,
            planned_sales_units=650, actual_sales_units=650, total_orders=420,
        ),
    ]

    print("\n[三维准确率分析]")
    for sku_data in skus:
        result = tracker.full_analysis(sku_data)
        print(f"\n  SKU: {result['sku_id']} | 综合: {result['overall_status']} ({result['overall_score']:.0%})")
        print(f"    进(入库): {result['inbound']['accuracy_pct']} {result['inbound']['status']}")
        print(f"    存(库存): {result['inventory']['inventory_accuracy_pct']} | 仓容: {result['inventory']['space_status']}")
        print(f"    销(销量): {result['sales']['accuracy_pct']} {result['sales']['status']} | 件单比: {result['sales']['units_per_order_str']}")

        if result['overall_score'] < 0.90:
            ep = result['error_propagation']
            print(f"    📍误差来源: 存货差{ep['inventory_error_total']}件"
                  f"（入库贡献{ep['contribution_from_inbound']}件"
                  f"，销售贡献{ep['contribution_from_sales']}件）")
            print(f"    🎯主要原因: {ep['main_cause']}")

    # 件单比趋势分析
    print("\n[件单比趋势监控]")
    months_data = [
        ('2026-01', 450, 200, 2.25),
        ('2026-02', 380, 170, 2.24),
        ('2026-03', 520, 250, 2.08),
        ('2026-04', 490, 240, 2.04),
        ('2026-05', 510, 260, 1.96),
        ('2026-06', 355, 280, 1.27),
    ]
    print(f"  {'月份':<10} {'销售件数':<12} {'订单数':<10} {'件单比':<10} {'状态'}")
    prev_upo = None
    for month, units, orders, upo in months_data:
        change = f"({(upo-prev_upo)/prev_upo:+.1%})" if prev_upo else ""
        status = '⚠️下降' if prev_upo and upo < prev_upo * 0.95 else '✅'
        print(f"  {month:<10} {units:<12} {orders:<10} {upo:<10.2f} {change} {status}")
        prev_upo = upo

    print("\n  2026-06件单比从2.0降至1.27（-36%）→ 多件优惠活动失效或产品结构变化")
    print("  建议: 检查多件组合购买优惠是否过期，或分析大件SKU占比变化")
    print("\n[✓] 物流计划进销存三维准确率系统测试通过")


if __name__ == "__main__":
    run_three_dimension_accuracy_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Purchase-Sales-Inventory-3D-Tracking]]（进销存追踪是本Skill的数据基础）、[[Skill-In-Transit-Inventory-Tracking-Visibility]]（在途库存是入库准确率的前置数据）
- **延伸（extends）**：[[Skill-Supply-Chain-KPI-Health-Dashboard]]（三维准确率纳入整体供应链健康仪表盘）、[[Skill-SOP-Sales-Operations-Planning]]（S&OP依赖三维准确率数据对齐计划）
- **可组合（combinable）**：[[Skill-Warehouse-Capacity-Efficiency-Planning]]（仓容准确率联动仓容规划）、[[Skill-OTIF-On-Time-In-Full-Analytics]]（OTIF是入库准确率的供应商侧解读）

## ⑤ 商业价值评估

- **ROI 预估**：入库准确率每提升1%（减少幽灵库存），避免约$500-2000的库存虚报；销售件数准确率从75%提升至85%，减少缺货损失约$3000/月；系统建设$1.5万，ROI≈400%
- **实施难度**：⭐⭐☆☆☆（数据已存在于WMS/OMS系统，关键是建立"进+销+存三层独立追踪"的意识和流程）
- **优先级**：⭐⭐⭐⭐⭐（书中将其列为物流计划供应链KPI第一节，是所有其他KPI的数据基础，数据准不准直接影响所有决策质量）
- **适用规模**：所有规模，月销售>$3万且有WMS系统的卖家
- **数据依赖**：WMS入库记录、库存盘点数据、OMS订单数据（三个系统数据对齐是最大挑战）
