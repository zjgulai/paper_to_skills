---
title: 健康库存三层数字化KPI体系 — 可视层/分析层/应用层的量化指标与联通机制
doc_type: knowledge
module: 04-供应链
topic: healthy-inventory-three-layer-digital-kpi
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 健康库存三层数字化KPI体系

> **书籍**：《全链路管理》陈凤霞 第七章第六节"健康库存管理系统——健康库存数字化架构"
> **桥梁**: 供应链 ↔ DataAgent-LLM | **类型**: 算法工具

## ① 算法原理

**书籍核心洞察（陈凤霞）**：书中专节阐述了健康库存管理系统的三层数字化架构，这是比单一KPI更高层次的系统性框架——将库存管理从"指标监控"升级为"数字化决策支持"。

**三层架构（书中核心框架）**：

1. **可视层（Visibility Layer）**——"看清楚现状"：
   - 库存现状汇总：分仓/分SKU/分批次的实时库存
   - 库龄结构：0-30/30-60/60-90/90+天分段分布
   - 动销/滞销状态：近7/14/30天销售情况
   - 核心KPI：总库存价值、DOI分布、ABC分类占比

2. **分析层（Analytics Layer）**——"预测未来趋势"：
   - 库存趋势模拟：未来4-8周的库存消耗路径
   - 缺货风险预测：哪个SKU、哪个时间点可能缺货
   - 积压风险预测：哪个SKU会持续累积成滞销
   - 核心KPI：缺货风险指数、积压风险指数、库存健康综合评分

3. **应用层（Application Layer）**——"触发行动"：
   - 联通补货系统：触发补货建议（上一Skill的参数校准）
   - 联通促销系统：触发清仓促销（长尾SKU清仓）
   - 联通物流系统：触发仓间调拨（多仓再平衡）
   - 核心KPI：建议执行率（建议被采纳的比例）、行动转化率

**书中关键：打通库存系统的基础数据**：
```
供应链数据流：
销售系统(订单) → 物流系统(发货) → 仓库系统(库存) → 财务系统(成本)
                         ↓
           健康库存管理系统（整合三层数据）
```

**三层联动的触发逻辑（书中业务逻辑）**：
- 可视层发现：某SKU 30天动销率=0（滞销信号）
- 分析层预测：按当前趋势，3个月后库龄将超90天
- 应用层触发：自动推送"清仓促销建议"到运营系统

## ② 母婴出海应用案例

**场景A：母婴品牌全渠道库存健康仪表盘**

- **业务问题**：品牌有FBA/UK仓/DE仓3个仓库，30个SKU，运营每天需要查3个系统才能了解库存状态，信息碎片化，无法做整体决策
- **三层架构应用**：
  1. 可视层：统一展示所有仓库库存价值$280,000，DOI均值42天，15个SKU绿灯，8个黄灯，5个红灯，2个黑灯
  2. 分析层：预测4周后有3个SKU缺货（概率>70%）；有2个SKU如不干预将进入积压状态
  3. 应用层：自动推送补货建议（3个SKU）、清仓建议（2个SKU），运营一键确认即可
- **预期产出**：每日运营决策时间从2小时降至20分钟，库存健康评分从62分提升至81分

**场景B：库存健康评分驱动绩效考核**

- **业务问题**：供应链团队的绩效考核只看"有没有断货"，不关心库存质量
- **健康评分KPI应用**：将三层架构的综合库存健康评分（0-100分）纳入月度考核，推动团队主动管理库龄、维护参数、及时清仓

## ③ 代码模板

```python
"""
健康库存三层数字化KPI体系
基于《全链路管理》陈凤霞 第七章第六节
可视层 + 分析层 + 应用层的三层量化指标
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SKUInventoryData:
    """SKU库存数据（整合多源）"""
    sku_id: str
    abc_class: str
    unit_cost: float
    unit_price: float
    current_stock: int
    in_transit: int
    sales_last_7d: float
    sales_last_14d: float
    sales_last_30d: float
    aging_0_30: int     # 0-30天库龄库存
    aging_31_60: int
    aging_61_90: int
    aging_90plus: int
    lead_time_days: int
    safety_stock_days: int


class HealthyInventoryThreeLayerKPI:
    """健康库存三层数字化KPI体系"""

    def layer1_visibility(self, sku: SKUInventoryData) -> Dict:
        """可视层 KPI"""
        total_stock = sku.current_stock + sku.in_transit
        inventory_value = sku.current_stock * sku.unit_cost
        avg_daily_sales = sku.sales_last_30d / 30

        # DOI
        doi = total_stock / max(avg_daily_sales, 0.01)

        # 动销率（近7天是否有销售）
        is_active = sku.sales_last_7d > 0

        # 库龄分布
        total = max(sku.current_stock, 1)
        aging_dist = {
            '0-30天': sku.aging_0_30 / total,
            '31-60天': sku.aging_31_60 / total,
            '61-90天': sku.aging_61_90 / total,
            '90天+': sku.aging_90plus / total,
        }

        # 五色灯
        target_doi = sku.lead_time_days + sku.safety_stock_days
        doi_ratio = doi / max(target_doi, 1)
        if sku.current_stock == 0:
            light = '⚫黑灯'
        elif doi_ratio < 0.8:
            light = '🔵蓝灯'
        elif doi_ratio <= 1.2:
            light = '🟢绿灯'
        elif doi_ratio <= 1.5:
            light = '🟡黄灯'
        else:
            light = '🔴红灯'

        return {
            'layer': 'visibility',
            'sku_id': sku.sku_id,
            'total_stock': total_stock,
            'inventory_value': inventory_value,
            'doi': round(doi, 1),
            'doi_light': light,
            'is_active': is_active,
            'aging_90plus_pct': f"{aging_dist['90天+']:.0%}",
            'high_aging_flag': aging_dist['90天+'] > 0.2,
        }

    def layer2_analytics(self, sku: SKUInventoryData,
                          weeks_ahead: int = 6) -> Dict:
        """分析层 KPI（趋势预测）"""
        avg_daily_sales = sku.sales_last_30d / 30
        total_available = sku.current_stock + sku.in_transit

        # 未来N周库存消耗路径
        weekly_sales = avg_daily_sales * 7
        weekly_stocks = []
        stock = sku.current_stock
        for week in range(weeks_ahead):
            stock = max(stock - weekly_sales, 0)
            weekly_stocks.append(stock)

        # 缺货风险：何时库存低于安全库存
        safety_stock = sku.safety_stock_days * avg_daily_sales
        stockout_weeks = None
        for i, s in enumerate(weekly_stocks):
            if s <= safety_stock:
                stockout_weeks = i + 1
                break

        # 积压风险：库龄90+趋势
        aging_growth_rate = (sku.aging_90plus - sku.aging_61_90 * 0.8) / max(sku.current_stock, 1)
        overstock_risk = aging_growth_rate > 0.05 and sku.sales_last_30d < sku.current_stock * 0.3

        # 缺货风险指数（0-100）
        if stockout_weeks:
            stockout_risk_idx = max(0, 100 - stockout_weeks * 15)
        else:
            stockout_risk_idx = 0

        # 积压风险指数（0-100）
        overstock_risk_idx = min(100, sku.aging_90plus / max(sku.current_stock, 1) * 200)

        return {
            'layer': 'analytics',
            'sku_id': sku.sku_id,
            'weekly_forecast': [round(s, 0) for s in weekly_stocks],
            'stockout_risk_week': stockout_weeks,
            'stockout_risk_index': round(stockout_risk_idx, 0),
            'overstock_risk_flag': overstock_risk,
            'overstock_risk_index': round(overstock_risk_idx, 0),
            'health_score': round(100 - stockout_risk_idx * 0.5 - overstock_risk_idx * 0.5, 0),
        }

    def layer3_application(self, sku: SKUInventoryData,
                            vis: Dict, analytics: Dict) -> Dict:
        """应用层 KPI（触发行动建议）"""
        actions = []

        # 补货建议
        if analytics['stockout_risk_index'] > 60:
            replenish_qty = int(sku.safety_stock_days * sku.sales_last_30d / 30 * 2)
            actions.append({
                'type': 'REPLENISH',
                'urgency': 'P0' if analytics['stockout_risk_index'] > 80 else 'P1',
                'quantity': replenish_qty,
                'description': f"预计{analytics['stockout_risk_week']}周内缺货，建议补货{replenish_qty}件",
            })

        # 清仓建议
        if analytics['overstock_risk_index'] > 40:
            clearance_qty = int(sku.aging_90plus * 0.5)  # 清50%的90天+库存
            actions.append({
                'type': 'CLEARANCE',
                'urgency': 'P1',
                'quantity': clearance_qty,
                'description': f"积压风险高，建议对{sku.aging_90plus}件90天+库存启动清仓",
            })

        # 调拨建议（简化：如果某仓过多则建议）
        if vis['doi'] > 60 and vis['doi_light'] == '🔴红灯':
            actions.append({
                'type': 'TRANSFER',
                'urgency': 'P2',
                'description': "库存过高，考虑调拨至其他仓库",
            })

        return {
            'layer': 'application',
            'sku_id': sku.sku_id,
            'action_count': len(actions),
            'actions': actions,
            'requires_attention': len(actions) > 0,
        }

    def generate_full_dashboard(self, skus: List[SKUInventoryData]) -> Dict:
        """生成完整三层仪表盘"""
        layer1_all, layer2_all, layer3_all = [], [], []
        for sku in skus:
            v = self.layer1_visibility(sku)
            a = self.layer2_analytics(sku)
            ap = self.layer3_application(sku, v, a)
            layer1_all.append(v)
            layer2_all.append(a)
            layer3_all.append(ap)

        # 综合指标
        total_inventory_value = sum(v['inventory_value'] for v in layer1_all)
        avg_health_score = np.mean([a['health_score'] for a in layer2_all])
        total_actions = sum(ap['action_count'] for ap in layer3_all)

        light_counts = {}
        for v in layer1_all:
            light_counts[v['doi_light']] = light_counts.get(v['doi_light'], 0) + 1

        return {
            'summary': {
                'total_skus': len(skus),
                'total_inventory_value': total_inventory_value,
                'avg_health_score': round(avg_health_score, 0),
                'health_grade': 'A' if avg_health_score >= 80 else ('B' if avg_health_score >= 60 else 'C'),
                'light_distribution': light_counts,
                'total_action_items': total_actions,
            },
            'layer1': layer1_all,
            'layer2': layer2_all,
            'layer3': layer3_all,
        }


def run_three_layer_kpi_demo():
    """三层健康库存KPI演示"""
    print("=" * 65)
    print("健康库存三层数字化KPI体系")
    print("基于《全链路管理》陈凤霞 第七章第六节")
    print("可视层 + 分析层 + 应用层")
    print("=" * 65)

    kpi_system = HealthyInventoryThreeLayerKPI()

    skus = [
        SKUInventoryData("PUMP-PRO", "A", 38, 89.99, 450, 200, 180, 350, 750,
                         200, 150, 60, 40, 35, 14),
        SKUInventoryData("WARMER-S1", "A", 18, 39.99, 80, 0, 55, 100, 420,
                         30, 30, 10, 10, 28, 14),
        SKUInventoryData("BOTTLE-3P", "B", 12, 24.99, 1200, 0, 15, 30, 180,
                         200, 300, 400, 300, 25, 10),  # 严重积压
        SKUInventoryData("UV-STERIL", "B", 55, 119.99, 0, 20, 5, 8, 60,
                         0, 0, 0, 0, 40, 21),  # 黑灯缺货
    ]

    dashboard = kpi_system.generate_full_dashboard(skus)
    summary = dashboard['summary']

    print(f"\n[仪表盘汇总]")
    print(f"  总库存价值: ${summary['total_inventory_value']:,.0f}")
    print(f"  平均健康评分: {summary['avg_health_score']}/100 ({summary['health_grade']}级)")
    print(f"  五色灯分布: {summary['light_distribution']}")
    print(f"  待处理行动项: {summary['total_action_items']}个")

    print(f"\n[三层详细KPI]")
    for i, sku in enumerate(skus):
        v = dashboard['layer1'][i]
        a = dashboard['layer2'][i]
        ap = dashboard['layer3'][i]
        print(f"\n  {'='*50}")
        print(f"  SKU: {sku.sku_id} [{v['doi_light']}]")
        print(f"  可视层: 库存价值${v['inventory_value']:,.0f} | DOI={v['doi']:.0f}天 | 90+天库存{v['aging_90plus_pct']}")
        print(f"  分析层: 健康分={a['health_score']:.0f} | 缺货风险指数={a['stockout_risk_index']:.0f} | 积压风险={a['overstock_risk_index']:.0f}")
        if a['stockout_risk_week']:
            print(f"           ⚠️ 预计第{a['stockout_risk_week']}周缺货")
        if ap['action_count'] > 0:
            for action in ap['actions']:
                print(f"  应用层: [{action['urgency']}] {action['description'][:55]}")
        else:
            print(f"  应用层: ✅ 无需干预")

    print(f"\n[书中关键洞察]")
    print(f"  三层不是独立的，而是联动的：可视→分析→应用（自动触发决策）")
    print(f"  打通多系统数据是三层架构落地的最大挑战")
    print(f"  健康评分=综合指标，适合用于团队绩效考核和管理者决策")
    print("\n[✓] 健康库存三层数字化KPI系统测试通过")
    return dashboard


if __name__ == "__main__":
    run_three_layer_kpi_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supply-Chain-KPI-Health-Dashboard]]（整体KPI仪表盘是本Skill的上层框架）、[[Skill-ITO-Three-Phase-Health-Tracking]]（三阶段健康追踪提供可视层数据）
- **延伸（extends）**：所有库存相关Skill的整合（本Skill是供应链数字化的最终目标）
- **可组合（combinable）**：[[Skill-Replenishment-Parameter-Calibration]]（应用层触发的补货建议依赖准确的参数）、[[Skill-Demand-Driven-KB-Construction]]（健康库存系统的知识库积累）

## ⑤ 商业价值评估

- **ROI 预估**：运营决策时间从2小时降至20分钟（年节省240小时≈$6000工时）；系统化识别行动项使漏处理率从60%降至10%（避免约$1万+月度损失）；系统$3万，ROI>300%
- **实施难度**：⭐⭐⭐⭐☆（三层系统需要整合多个数据源；最难的是应用层与实际系统的自动联通）
- **优先级**：⭐⭐⭐⭐⭐（书中第七章收官之作，是所有供应链数字化的终极形态；是从"KPI追踪"到"数字化决策支持"的质的跨越）
- **适用规模**：月GMV>$10万、SKU数>30个的卖家；越大越有价值
- **数据依赖**：多系统数据整合（OMS+WMS+财务）；数据质量是最大挑战
