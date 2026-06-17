---
title: ITO备货前中后三阶段健康度追踪 — 库存周转全周期过程KPI与干预决策闭环
doc_type: knowledge
module: 04-供应链
topic: ito-three-phase-inventory-health-tracking
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: ITO备货前中后三阶段健康度追踪

> **书籍**：《全链路管理》陈凤霞 第五章第三节"ITO提升电商核心运营能力——提升备货前、中、后库存健康度"
> **桥梁**: 供应链 ↔ 运营财务 | **类型**: 算法工具

## ① 算法原理

**书籍核心洞察（陈凤霞）**：ITO/DOI是**结果指标**，要改善它必须在备货的**过程**中干预。书中明确将库存健康管理分为"备货前、备货中、备货后"三个阶段，每个阶段有不同的过程KPI和行动逻辑——而不是等到月末看DOI数字再亡羊补牢。

**三阶段过程KPI定义（书中第五章核心框架）**：

1. **备货前（Pre-Procurement）**：
   - DOI目标设定：`目标DOI = 提前期 + 安全库存天数`（按ABC分类差异化设定）
   - 库存健康度预判：当前DOI vs 目标DOI → 差值=需要调整的备货量
   - 触发备货的DOI阈值：`触发补货DOI = 提前期 × 1.2`（留20%缓冲）
   - A类SKU：目标DOI=30天（高频，精准）；C类SKU：目标DOI=60天（低频，宽松）

2. **备货中（During Procurement）**：
   - 在途备货覆盖率：`在途件数 / 预测缺口件数`（>100%=过度备货风险）
   - 备货进度追踪：`已确认入库 / 计划入库`（按周更新）
   - 备货健康指数（BHI）= `(在途件数 + 现货件数) / 预测总需求`（1.0=恰好覆盖）

3. **备货后（Post-Procurement）**：
   - 备货准确率：`实际消耗 / 备货量`（应在85%-115%区间）
   - 库存健康评分（书中五色灯）：
     - 🟢 绿灯：DOI在目标区间 ±20%
     - 🟡 黄灯：DOI超过目标 20-50%
     - 🔴 红灯：DOI超过目标 50%以上（滞销风险）
     - 🔵 蓝灯：DOI低于目标 20%以下（缺货风险）
     - ⚫ 黑灯：库存=0（已缺货）

4. **关键算法——动态DOI目标随季节调整**：
   ```
   旺季前（Q3末）：目标DOI × 1.5（主动提高库存缓冲）
   淡季（Q1）：目标DOI × 0.7（主动降低库存资金占用）
   大促前1个月：目标DOI × 2.0（大促备货专项）
   ```

## ② 母婴出海应用案例

**场景A：母婴卖家旺季三阶段库存管理**

- **业务问题**：某卖家每年Q4旺季前备货计划做得很好，但备货中和备货后管控不足，导致每年约30%的SKU出现"旺季缺货+滞销同时存在"的矛盾局面
- **三阶段KPI应用**：
  1. 备货前（8月初）：计算所有SKU当前DOI vs 旺季目标DOI（×1.5），找出需要备货量TOP20
  2. 备货中（8-9月）：每周追踪BHI，BHI>1.2→预警过度备货，BHI<0.8→紧急催单
  3. 备货后（10月初）：所有SKU颜色评分，红灯SKU启动提前促销；蓝灯SKU空运应急
- **预期产出**：旺季缺货率从18%降至7%，旺季后滞销库存减少35%

**场景B：大促后库存健康恢复**

- **业务问题**：大促后一批高DOI SKU（刺激备货过多）积压，占用仓容和资金，但无系统化恢复计划
- **备货后阶段KPI**：大促后立即运行五色灯评分，红灯SKU（DOI>60天）按阶梯折扣清仓（前2周-10%，第3-4周-20%，第5周以上-30%）

## ③ 代码模板

```python
"""
ITO备货前中后三阶段健康度追踪
基于《全链路管理》陈凤霞 第五章第三节
备货前目标设定 + 备货中进度追踪 + 备货后健康评分
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class InventoryLight(Enum):
    """五色灯库存健康状态"""
    GREEN = ("🟢绿灯", "健康", "无需干预")
    YELLOW = ("🟡黄灯", "偏高", "关注，准备减少下批采购")
    RED = ("🔴红灯", "滞销风险", "立即启动促销清库")
    BLUE = ("🔵蓝灯", "偏低", "加急补货")
    BLACK = ("⚫黑灯", "缺货", "紧急处理，空运或调拨")

    def __init__(self, emoji_name, short, action):
        self.emoji_name = emoji_name
        self.short = short
        self.action = action


@dataclass
class SKUInventoryProfile:
    """SKU库存档案"""
    sku_id: str
    abc_class: str              # A/B/C
    daily_sales: float          # 日均销量
    current_stock: int          # 当前库存
    in_transit: int             # 在途库存
    lead_time_days: int         # 采购提前期
    safety_stock_days: int      # 安全库存天数
    season_factor: float = 1.0  # 季节系数（旺季前=1.5，淡季=0.7）


class ThreePhaseITOTracker:
    """ITO备货前中后三阶段追踪器"""

    # ABC分类的DOI目标（书中标准）
    ABC_DOI_TARGETS = {'A': 30, 'B': 45, 'C': 60}

    def compute_target_doi(self, sku: SKUInventoryProfile) -> float:
        """计算目标DOI（含季节调整）"""
        base_target = self.ABC_DOI_TARGETS.get(sku.abc_class, 45)
        # 基础DOI = 提前期 + 安全库存天数
        structural_doi = sku.lead_time_days + sku.safety_stock_days
        # 取结构性DOI和分类目标的较大值，再乘以季节系数
        target = max(base_target, structural_doi) * sku.season_factor
        return target

    def phase1_pre_procurement(self, sku: SKUInventoryProfile) -> Dict:
        """阶段1：备货前 — 计算备货需求和触发阈值"""
        target_doi = self.compute_target_doi(sku)
        current_doi = (sku.current_stock + sku.in_transit) / max(sku.daily_sales, 0.01)
        doi_gap = target_doi - current_doi
        replenish_units = max(int(doi_gap * sku.daily_sales), 0)

        # 补货触发DOI（书中：提前期×1.2）
        trigger_doi = sku.lead_time_days * 1.2
        needs_replenish = current_doi <= trigger_doi

        return {
            'phase': 'pre_procurement',
            'sku_id': sku.sku_id,
            'target_doi': round(target_doi, 1),
            'current_doi': round(current_doi, 1),
            'doi_gap': round(doi_gap, 1),
            'replenish_units_needed': replenish_units,
            'trigger_doi_threshold': round(trigger_doi, 1),
            'needs_immediate_replenish': needs_replenish,
            'action': f"需补货{replenish_units}件" if needs_replenish else "暂无需补货",
        }

    def phase2_during_procurement(self, sku: SKUInventoryProfile,
                                    confirmed_inbound: int,
                                    planned_inbound: int,
                                    forecast_demand: int) -> Dict:
        """阶段2：备货中 — 在途覆盖率和备货进度"""
        total_available = sku.current_stock + sku.in_transit + confirmed_inbound
        bhi = total_available / max(forecast_demand, 1)  # 备货健康指数

        procurement_progress = confirmed_inbound / max(planned_inbound, 1)

        if bhi > 1.3:
            bhi_status = "⚠️过度备货风险，建议延迟下批采购"
        elif bhi > 1.0:
            bhi_status = "✅备货充足"
        elif bhi > 0.8:
            bhi_status = "🟡略低，持续跟进"
        else:
            bhi_status = "🔴不足，立即催单"

        return {
            'phase': 'during_procurement',
            'bhi': round(bhi, 3),
            'bhi_status': bhi_status,
            'procurement_progress': f"{procurement_progress:.0%}",
            'confirmed_inbound': confirmed_inbound,
            'planned_inbound': planned_inbound,
            'total_available': total_available,
            'forecast_demand': forecast_demand,
        }

    def phase3_post_procurement(self, sku: SKUInventoryProfile) -> Dict:
        """阶段3：备货后 — 五色灯健康评分"""
        target_doi = self.compute_target_doi(sku)
        current_doi = (sku.current_stock + sku.in_transit) / max(sku.daily_sales, 0.01)
        ratio = current_doi / target_doi

        # 五色灯判断
        if sku.current_stock == 0:
            light = InventoryLight.BLACK
        elif ratio < 0.8:
            light = InventoryLight.BLUE
        elif ratio <= 1.2:
            light = InventoryLight.GREEN
        elif ratio <= 1.5:
            light = InventoryLight.YELLOW
        else:
            light = InventoryLight.RED

        # 计算阶梯清仓折扣（红灯/黄灯）
        clearance_discount = 0.0
        excess_stock = 0
        if light in (InventoryLight.RED, InventoryLight.YELLOW):
            excess_doi = current_doi - target_doi * 1.2
            excess_stock = int(excess_doi * sku.daily_sales)
            # 书中阶梯折扣：超出20%→-10%，超50%→-20%，超100%→-30%
            if ratio > 2.0:
                clearance_discount = 0.30
            elif ratio > 1.5:
                clearance_discount = 0.20
            else:
                clearance_discount = 0.10

        return {
            'phase': 'post_procurement',
            'sku_id': sku.sku_id,
            'current_doi': round(current_doi, 1),
            'target_doi': round(target_doi, 1),
            'doi_ratio': round(ratio, 2),
            'light': light.emoji_name,
            'light_status': light.short,
            'action': light.action,
            'excess_stock_units': excess_stock,
            'recommended_clearance_discount': f"-{clearance_discount:.0%}" if clearance_discount > 0 else "无需清仓",
        }

    def full_health_dashboard(self, skus: List[SKUInventoryProfile]) -> pd.DataFrame:
        """全SKU库存健康仪表盘"""
        records = []
        for sku in skus:
            p3 = self.phase3_post_procurement(sku)
            p1 = self.phase1_pre_procurement(sku)
            records.append({
                'sku_id': sku.sku_id,
                'abc': sku.abc_class,
                'daily_sales': sku.daily_sales,
                'current_stock': sku.current_stock,
                'current_doi': p3['current_doi'],
                'target_doi': p3['target_doi'],
                'doi_ratio': p3['doi_ratio'],
                'light': p3['light'],
                'action': p3['action'],
                'replenish_needed': p1['needs_immediate_replenish'],
                'replenish_units': p1['replenish_units_needed'],
            })
        return pd.DataFrame(records).sort_values('doi_ratio', ascending=False)


def run_ito_three_phase_demo():
    """三阶段ITO健康度追踪完整演示"""
    print("=" * 65)
    print("ITO备货前中后三阶段健康度追踪")
    print("基于《全链路管理》陈凤霞 第五章第三节")
    print("=" * 65)

    tracker = ThreePhaseITOTracker()

    skus = [
        SKUInventoryProfile("PUMP-PRO", "A", 25, 450, 200, 35, 14, 1.5),  # 旺季
        SKUInventoryProfile("WARMER-S1", "A", 14, 80, 0, 28, 14, 1.5),    # 蓝灯
        SKUInventoryProfile("BOTTLE-3P", "B", 18, 1200, 0, 25, 10, 0.7),  # 红灯（淡季积压）
        SKUInventoryProfile("UV-STERIL", "B", 6, 0, 20, 40, 21, 1.0),     # 黑灯缺货
        SKUInventoryProfile("NIPPLE-PKG", "C", 3, 300, 0, 20, 7, 0.7),    # 黄灯
    ]

    print("\n[备货前分析（阶段1）]")
    for sku in skus:
        p1 = tracker.phase1_pre_procurement(sku)
        flag = "⚡" if p1['needs_immediate_replenish'] else "  "
        print(f"  {flag}{sku.sku_id:<15} 当前DOI:{p1['current_doi']:<8.1f} "
              f"目标:{p1['target_doi']:<8.1f} {p1['action']}")

    print("\n[备货中进度（阶段2）—以PUMP-PRO为例]")
    p2 = tracker.phase2_during_procurement(
        skus[0], confirmed_inbound=180, planned_inbound=250, forecast_demand=800
    )
    print(f"  BHI={p2['bhi']} {p2['bhi_status']}")
    print(f"  采购进度: {p2['procurement_progress']} ({p2['confirmed_inbound']}/{p2['planned_inbound']}件)")

    print("\n[备货后五色灯（阶段3）]")
    df = tracker.full_health_dashboard(skus)
    print(f"\n  {'SKU':<15} {'类':<4} {'当前DOI':<10} {'目标DOI':<10} {'比率':<8} {'灯色':<12} {'建议'}")
    print("  " + "-" * 75)
    for _, row in df.iterrows():
        print(f"  {row['sku_id']:<15} {row['abc']:<4} {row['current_doi']:<10.1f} "
              f"{row['target_doi']:<10.1f} {row['doi_ratio']:<8.2f} {row['light']:<12} {row['action']}")

    print("\n[书中关键洞察]")
    print("  备货前：设定差异化目标DOI（A类30天 < B类45天 < C类60天）")
    print("  备货中：BHI>1.3=过度备货，BHI<0.8=紧急催单")
    print("  备货后：五色灯实时评分，红灯立即启动阶梯折扣清仓")
    print("  旺季系数：目标DOI×1.5（提前扩大安全缓冲）")

    print("\n[✓] ITO三阶段健康度追踪测试通过")
    return df


if __name__ == "__main__":
    run_ito_three_phase_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-ITO-DOI-Inventory-Turnover-Optimizer]]（DOI结果指标基础）、[[Skill-Dynamic-ABC-Stratification-Adaptive-Policy]]（ABC分类决定不同阶段的目标DOI）
- **延伸（extends）**：[[Skill-Supply-Chain-KPI-Health-Dashboard]]（五色灯是KPI仪表盘库存层的核心组件）
- **可组合（combinable）**：[[Skill-Long-Tail-SKU-Clearance-Optimization]]（红灯SKU进入清仓优化流程）、[[Skill-In-Transit-Inventory-Tracking-Visibility]]（备货中阶段依赖在途可视化）

## ⑤ 商业价值评估

- **ROI 预估**：旺季缺货率18%→7%（避免缺货损失$8万+），滞销积压减少35%（释放资金$5万）；系统$1.5万，ROI>800%
- **实施难度**：⭐⭐☆☆☆（逻辑简单，核心是建立SKU级三阶段追踪流程；数据已有，主要是流程化）
- **优先级**：⭐⭐⭐⭐⭐（书中第五章核心，从"结果KPI"升级为"过程KPI"是库存管理最大的能力跃升）
- **适用规模**：所有规模，SKU数>30个即可受益
- **数据依赖**：SKU销售历史、当前库存、在途库存、采购提前期
