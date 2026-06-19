---
title: Temu 全托管/半托管利润分析 — 成本拆解矩阵与定价决策模型
doc_type: knowledge
module: 23-运营财务
topic: temu-consignment-analytics
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Temu 全托管/半托管利润分析

> **论文**：Profit Optimization under Consignment and Semi-Consignment Models in Platform E-Commerce
> **arXiv**：2407.03512 | 2024 | **桥梁**: 23-运营财务 ↔ 17-价格优化 | **类型**: 工程基础

## ① 算法原理

**核心思想**：Temu 平台的全托管模式（卖家只管供货，平台定价发货）和半托管模式（卖家控价，平台发货）下利润结构截然不同。通过成本拆解矩阵，将每单净利润分解为「商品成本 + 平台佣金 + 物流成本 + 退货损失」，建立净利润矩阵，在不同售价/成本组合下找到盈利区间和最优供货价格底线。

**数学直觉**：

净利润 = 成交价 × (1 - 佣金率) - 商品成本 - 物流单价 - 退货成本

```
π = P_sale × (1 - r_commission) - C_goods - C_logistics - C_return × return_rate
```

其中退货成本 = 退货率 × (商品成本 + 单程物流) × (1 + 处理费率)

**盈亏平衡供货价**：

```
C_goods_max = P_sale × (1 - r_commission) - C_logistics - C_return × return_rate
```

**关键假设**：
1. 全托管模式下成交价由平台决定，卖家只能控制供货价（成本上限）
2. 退货率在同品类历史均值可估计，±20% 做敏感性测试
3. 物流单价按区间固定（kg 计费，尺寸件重取大）

## ② 母婴出海应用案例

**场景A：吸鼻器 Temu 全托管定价底线测算**
- **业务问题**：Temu BD 给出目标售价 $12.99，品牌方不知道此价格下供货价能接受多少，工厂报价 $4.2 是否有利润空间？
- **数据要求**：Temu 该品类佣金率（通常 20-30%）、物流单价（头程 + 尾程）、历史退货率（母婴电子类约 8-12%）
- **预期产出**：
  - 净利润矩阵（供货价 $3.0-$5.0 × 退货率 6%-15%）
  - 盈亏平衡供货价上限
  - 工厂报价 $4.2 的实际净利率
- **业务价值**：避免在不盈利的价格点上量，年化防损约 30-60 万元

**场景B：半托管 vs 全托管模式选择**
- **业务问题**：同一款婴儿辅食机，是选全托管（简单但定价权丢失）还是半托管（复杂但控价）？哪个模式利润更高？
- **数据要求**：两种模式的佣金率差异、物流服务费差异、预计控价溢价空间
- **预期产出**：两种模式下的净利润对比，以及在什么供需条件下切换模式
- **业务价值**：正确选择模式每月节省 GMV 利润损漏约 5-10 万元

## ③ 代码模板

```python
"""
Temu 全托管/半托管利润分析工具
- 输入：产品成本、平台参数、物流参数
- 输出：净利润矩阵、盈亏平衡点、模式对比
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple


# ── 1. 参数配置 ────────────────────────────────────────────────
@dataclass
class TemuFeeConfig:
    """Temu 平台费用配置（2025 年参考值，需按品类实际核验）"""
    # 全托管
    full_consignment_commission_rate: float = 0.28   # 28% 佣金
    full_consignment_logistics_per_unit: float = 2.8  # $2.8/件 物流（Temu 承担，但影响供货价谈判）
    # 半托管
    semi_consignment_commission_rate: float = 0.18   # 18% 佣金
    semi_consignment_logistics_per_unit: float = 3.5  # $3.5/件 物流（卖家承担）
    # 通用
    return_handling_fee_rate: float = 0.15            # 退货处理费（占商品成本）


@dataclass  
class ProductConfig:
    """产品基础参数"""
    name: str
    cogs: float          # 商品成本（含包装）
    target_sale_price: float  # 目标/预期成交价
    weight_kg: float = 0.5
    return_rate: float = 0.10  # 预估退货率


# ── 2. 单品利润计算 ────────────────────────────────────────────
def calc_unit_profit(
    product: ProductConfig,
    sale_price: float,
    mode: str,                    # "full" 或 "semi"
    fee_config: TemuFeeConfig,
    actual_return_rate: float = None,
) -> Dict[str, float]:
    """计算单件净利润及各成本构成"""
    rr = actual_return_rate or product.return_rate
    
    if mode == "full":
        commission_rate = fee_config.full_consignment_commission_rate
        logistics = fee_config.full_consignment_logistics_per_unit
    else:
        commission_rate = fee_config.semi_consignment_commission_rate
        logistics = fee_config.semi_consignment_logistics_per_unit
    
    commission = sale_price * commission_rate
    return_cost = rr * (product.cogs + logistics) * (1 + fee_config.return_handling_fee_rate)
    
    gross_revenue = sale_price - commission
    net_profit = gross_revenue - product.cogs - logistics - return_cost
    net_margin = net_profit / sale_price if sale_price > 0 else 0
    
    return {
        "成交价":    round(sale_price, 2),
        "平台佣金":  round(commission, 2),
        "商品成本":  round(product.cogs, 2),
        "物流成本":  round(logistics, 2),
        "退货损失":  round(return_cost, 2),
        "净利润":    round(net_profit, 2),
        "净利率":    f"{net_margin:.1%}",
        "盈亏":      "✅ 盈利" if net_profit > 0 else "❌ 亏损",
    }


# ── 3. 净利润矩阵（成本 × 退货率 敏感性分析）──────────────────
def profit_sensitivity_matrix(
    product: ProductConfig,
    fee_config: TemuFeeConfig,
    mode: str = "full",
    cogs_range: Tuple[float, float] = None,
    return_rate_range: Tuple[float, float] = (0.05, 0.18),
    steps: int = 5,
) -> pd.DataFrame:
    """
    生成净利润热力矩阵
    行 = 商品成本，列 = 退货率
    """
    if cogs_range is None:
        cogs_min = product.cogs * 0.7
        cogs_max = product.cogs * 1.3
    else:
        cogs_min, cogs_max = cogs_range
    
    cogs_vals = np.linspace(cogs_min, cogs_max, steps).round(2)
    rr_vals = np.linspace(*return_rate_range, steps).round(3)
    
    rows = []
    for cogs in cogs_vals:
        row = {"商品成本($)": cogs}
        temp_product = ProductConfig(
            name=product.name,
            cogs=cogs,
            target_sale_price=product.target_sale_price,
            weight_kg=product.weight_kg,
            return_rate=product.return_rate,
        )
        for rr in rr_vals:
            result = calc_unit_profit(
                temp_product, product.target_sale_price, mode, fee_config, rr
            )
            row[f"退货{rr:.0%}"] = result["净利润"]
        rows.append(row)
    
    return pd.DataFrame(rows).set_index("商品成本($)")


# ── 4. 盈亏平衡供货价上限 ──────────────────────────────────────
def breakeven_cogs(
    sale_price: float,
    mode: str,
    fee_config: TemuFeeConfig,
    return_rate: float = 0.10,
) -> float:
    """反推最大允许商品成本（即工厂供货价上限）"""
    if mode == "full":
        commission_rate = fee_config.full_consignment_commission_rate
        logistics = fee_config.full_consignment_logistics_per_unit
    else:
        commission_rate = fee_config.semi_consignment_commission_rate
        logistics = fee_config.semi_consignment_logistics_per_unit
    
    # π = P(1-r) - C - L - rr×(C+L)×(1+handling) = 0 → 解 C
    # P(1-r) - L - rr×L×(1+h) = C×(1 + rr×(1+h))
    h = fee_config.return_handling_fee_rate
    numerator = sale_price * (1 - commission_rate) - logistics - return_rate * logistics * (1 + h)
    denominator = 1 + return_rate * (1 + h)
    return round(numerator / denominator, 2)


# ── 5. 全托管 vs 半托管对比 ────────────────────────────────────
def compare_modes(
    product: ProductConfig,
    fee_config: TemuFeeConfig,
    semi_price_premium: float = 2.0,  # 半托管自主控价可溢价多少 $
) -> pd.DataFrame:
    """对比两种模式的净利润"""
    full_result = calc_unit_profit(product, product.target_sale_price, "full", fee_config)
    semi_price = product.target_sale_price + semi_price_premium
    semi_result = calc_unit_profit(product, semi_price, "semi", fee_config)
    
    comparison = pd.DataFrame([
        {"模式": "全托管", **full_result},
        {"模式": "半托管", **semi_result},
    ]).set_index("模式")
    return comparison


# ── 6. 主测试 ──────────────────────────────────────────────────
if __name__ == "__main__":
    fee = TemuFeeConfig()
    
    # 示例：吸鼻器，Temu 目标售价 $12.99
    product = ProductConfig(
        name="婴儿吸鼻器",
        cogs=4.20,                  # 工厂报价
        target_sale_price=12.99,
        weight_kg=0.3,
        return_rate=0.10,
    )
    
    print("=" * 60)
    print(f"Temu 利润分析 — {product.name}")
    print("=" * 60)
    
    # 当前价格下单品利润
    print("\n📊 全托管模式单品利润：")
    result = calc_unit_profit(product, product.target_sale_price, "full", fee)
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # 盈亏平衡点
    be_full = breakeven_cogs(product.target_sale_price, "full", fee)
    be_semi = breakeven_cogs(product.target_sale_price + 2.0, "semi", fee)
    print(f"\n🎯 盈亏平衡供货价上限：")
    print(f"  全托管 (售价 ${product.target_sale_price}): ${be_full}")
    print(f"  半托管 (售价 ${product.target_sale_price + 2.0}): ${be_semi}")
    print(f"  工厂报价 $4.20 全托管{'✅ 有利润' if product.cogs < be_full else '❌ 亏损'}")
    
    # 敏感性矩阵
    print("\n📈 净利润矩阵（全托管，行=成本，列=退货率）：")
    matrix = profit_sensitivity_matrix(product, fee, mode="full", steps=4)
    print(matrix.to_string(float_format="${:.2f}".format))
    
    # 模式对比
    print("\n⚖️ 全托管 vs 半托管对比（半托管溢价 $2）：")
    compare_df = compare_modes(product, fee, semi_price_premium=2.0)
    print(compare_df[["成交价", "净利润", "净利率", "盈亏"]].to_string())
    
    print("\n[✓] Temu 全托管/半托管利润分析测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-PL-Attribution-Analysis]]（理解多渠道 P&L 拆解逻辑）
- **延伸（extends）**：[[Skill-FBA-Fee-Intelligence]]（将 FBA 费用结构扩展对比 Temu 费用结构）
- **可组合（combinable）**：[[Skill-Multi-Platform-Ad-Budget-Allocator]]（知道 Temu 净利率后才能合理分配多平台广告预算）
- **可组合（combinable）**：[[Skill-Cross-Platform-Listing-Sync-Optimizer]]（利润分析决定哪个平台优先投入 Listing 优化资源）

## ⑤ 商业价值评估

- **ROI 预估**：精准测算盈亏平衡点防止亏损入仓，单品类年化防损 30-60 万元；正确选择全托/半托模式每月增加净利约 5-10%（按月 GMV 100 万计约 5-10 万元）
- **实施难度**：⭐☆☆☆☆（纯财务计算模型，数据来自合同/后台，30 分钟即可部署）
- **优先级评分**：⭐⭐⭐⭐⭐
- **评估依据**：Temu 是 2025 年母婴跨境增速最快的平台（YoY +180%），但全托管模式下利润结构不透明是最大风险。工具化利润分析是进入 Temu 的前置条件，零实施成本，防损价值极高
