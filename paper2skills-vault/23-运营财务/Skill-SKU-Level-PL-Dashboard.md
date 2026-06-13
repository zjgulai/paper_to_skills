---
title: SKU-Level PL Dashboard — 单品利润核算：每个 SKU 今天赚了多少钱
doc_type: knowledge
module: 23-运营财务
topic: sku-level-pl-dashboard
status: stable
created: 2026-06-12
updated: 2026-06-12
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: SKU-Level PL Dashboard — 单品 P&L 实时核算

> **方法**：FBA 成本结构分解 + 贡献毛利分析（Contribution Margin Analysis）
> **来源**：行业标准 FBA 成本公式 + arXiv 2211.09612（利润优化字段验证）
> **桥梁**: 23-运营财务 ↔ 13-广告分析 ↔ 04-供应链 | **类型**: 财务工具
> **反直觉来源**：23-运营财务有 14 个 Skill 但 avg_in=2.5（全图最低），根本原因是缺少"单品赚多少钱"这个最基础的财务工具

---

## ① 算法原理

### 核心思想

跨境电商的 P&L 核算有一个普遍性问题：财务报表在品牌/账号层面，但决策需要在 SKU 层面——不知道吸奶器 M5 今天赚了多少钱，就无法判断是否要继续备货、调整定价、增减广告。

**SKU-Level P&L 公式**：

```
单品贡献毛利 = 销售额
             - 产品成本（COGS）
             - FBA 拣货配送费（Fulfillment Fee）
             - FBA 月度仓储费（Storage Fee）
             - 广告归因花费（Ad Spend per Unit）
             - 退货损失（Return Rate × Return Cost）
             - Amazon 平台佣金（Referral Fee = 售价 × 8-15%）
             - FBA 入库费（Inbound Fee）
```

**三级 P&L 结构**：

```
售价（Gross Revenue）
   └─ 退款/退货         →  净销售额（Net Revenue）
      └─ COGS           →  毛利（Gross Profit）
         └─ FBA 费用    →  FBA 后毛利（Post-FBA Margin）
            └─ 广告费   →  广告后利润（Post-Ad Profit）
               └─ 仓储/入库  →  贡献毛利（Contribution Margin）
```

### FBA 费用计算

Amazon FBA 费用由产品尺寸和重量决定，2026 年标准件（6oz-12oz，8"×5"×3"以内）：

| 费用类型 | 2026 年费率 |
|---|---|
| 拣货配送费（Standard Size） | $3.06-$4.25/件 |
| 月度仓储费（Q1-Q3） | $0.78/立方英尺/月 |
| 月度仓储费（Q4旺季） | $2.40/立方英尺/月 |
| Amazon 平台佣金 | 8-15%（母婴品类 8%）|
| FBA 入库费 | $0.27-$1.58/件 |
| 退货处理费 | $2.45/件 |

### 关键洞察

最反直觉的发现是**广告费通常是利润最大的侵蚀项**：当 ACOS=35% 而毛利率=38% 时，广告贡献毛利接近于零——这个计算绝大多数团队没有按 SKU 做，导致"广告带来销售"的幻觉掩盖了"广告消灭利润"的现实。

---

## ② 母婴出海应用案例

### 场景 A：发现"隐性亏损 SKU"（数量最大的分析价值）

**业务问题**：账号整体毛利率 22%，看起来还行，但 CFO 发现实际现金流一直很紧。不知道哪些 SKU 在拖后腿。

**SKU P&L 分析结果**：

| SKU | 售价 | COGS | FBA 费 | 广告费/件 | 退货损失 | 贡献毛利 |
|---|---|---|---|---|---|---|
| M5 吸奶器 | $89.99 | $28 | $4.45 | $8.20 | $3.20 | **+$45.14 (50.2%)** |
| 储奶袋 50片 | $12.99 | $3.20 | $3.22 | $4.80 | $0.52 | **-$0.75 (亏损)** |
| UV 消毒器 | $59.99 | $22 | $5.80 | $7.50 | $2.40 | **+$19.29 (32.2%)** |

**发现**：储奶袋配件 SKU 每卖一件亏损 $0.75——因为广告费/件过高，而低单价吸收不了 FBA 固定费用。

**决策**：停止储奶袋 PPC 广告，改用"经常一起购买"关联推荐；或提价到 $14.99

### 场景 B：定价决策支持（提价 $5 的真实影响）

**业务问题**：M5 吸奶器想从 $89.99 提价到 $94.99，团队担心销量下降。财务计算提价 vs 不提价哪个更划算。

**P&L 对比**：假设提价后销量下降 8%（价格弹性 -1.6）：
- 不提价：100件 × $45.14 = $4,514 贡献毛利
- 提价后：92件 × $50.14 = $4,613 贡献毛利（+$99）

**结论**：提价在合理弹性范围内更有利，且可以减少 FBA 仓储压力。

---

## ③ 代码模板

```python
"""
SKU-Level P&L Dashboard — 单品利润核算模型
基于 Amazon FBA 成本结构 + Contribution Margin 分析

依赖: dataclasses, statistics (标准库)
"""

from dataclasses import dataclass, field
from statistics import mean
from typing import Optional


@dataclass
class FBAConfig:
    """FBA 费用配置（2026 年标准）"""
    referral_fee_rate: float = 0.08       # Amazon 平台佣金率（母婴品类 8%）
    fulfillment_fee: float = 4.25         # 拣货配送费（标准件）
    storage_rate_monthly: float = 0.78   # 月度仓储费（$0.78/立方英尺，Q1-Q3）
    storage_rate_q4: float = 2.40        # Q4 旺季仓储费
    inbound_fee_per_unit: float = 0.50   # FBA 入库费
    return_processing_fee: float = 2.45  # 退货处理费
    product_cubic_feet: float = 0.20     # 商品体积（立方英尺）


@dataclass
class SKUCosts:
    """单 SKU 成本结构"""
    sku_id: str
    title: str
    selling_price: float
    cogs: float                          # 产品成本（含采购+头程）
    return_rate: float = 0.06            # 退货率
    is_q4: bool = False                  # 是否 Q4 旺季（影响仓储费）
    ad_spend_per_unit: float = 0.0       # 广告费/件（= 广告总花费/出货量）
    monthly_units_sold: int = 100        # 月销售量


@dataclass
class SKUPL:
    """单 SKU P&L 报表"""
    sku_id: str
    title: str
    # 收入层
    gross_revenue_per_unit: float
    return_loss_per_unit: float
    net_revenue_per_unit: float
    # 成本层
    cogs: float
    gross_profit: float
    gross_margin: float
    # FBA 层
    referral_fee: float
    fulfillment_fee: float
    storage_fee: float
    inbound_fee: float
    total_fba_fee: float
    post_fba_profit: float
    post_fba_margin: float
    # 广告层
    ad_spend_per_unit: float
    post_ad_profit: float
    # 最终
    contribution_margin: float
    contribution_margin_rate: float
    # 月度汇总
    monthly_units: int
    monthly_contribution: float
    # 状态
    is_profitable: bool
    primary_cost_driver: str           # 最大成本项


class SKUPLCalculator:
    """单品 P&L 计算器"""

    def __init__(self, fba_config: FBAConfig = None):
        self.fba = fba_config or FBAConfig()

    def calculate(self, sku: SKUCosts) -> SKUPL:
        """计算单 SKU 完整 P&L"""
        p = sku.selling_price

        # 1. 收入层
        referral_fee = p * self.fba.referral_fee_rate
        return_loss = sku.return_rate * (p + self.fba.return_processing_fee)
        net_rev = p - return_loss

        # 2. 毛利层
        gross_profit = net_rev - sku.cogs - referral_fee
        gross_margin = gross_profit / p if p > 0 else 0

        # 3. FBA 费用层
        storage_rate = self.fba.storage_rate_q4 if sku.is_q4 else self.fba.storage_rate_monthly
        storage_fee = self.fba.product_cubic_feet * storage_rate
        total_fba = (self.fba.fulfillment_fee + storage_fee +
                     self.fba.inbound_fee_per_unit)
        post_fba = gross_profit - total_fba
        post_fba_margin = post_fba / p if p > 0 else 0

        # 4. 广告层
        post_ad = post_fba - sku.ad_spend_per_unit

        # 5. 贡献毛利（=广告后利润，已含所有可变成本）
        cm = post_ad
        cm_rate = cm / p if p > 0 else 0

        # 6. 找最大成本项
        costs = {
            "COGS":  sku.cogs,
            "广告费": sku.ad_spend_per_unit,
            "FBA费": total_fba,
            "退货":  return_loss,
            "平台佣金": referral_fee,
        }
        primary_driver = max(costs, key=costs.get)

        return SKUPL(
            sku_id=sku.sku_id,
            title=sku.title[:40],
            gross_revenue_per_unit=round(p, 2),
            return_loss_per_unit=round(return_loss, 2),
            net_revenue_per_unit=round(net_rev, 2),
            cogs=round(sku.cogs, 2),
            gross_profit=round(gross_profit, 2),
            gross_margin=round(gross_margin, 4),
            referral_fee=round(referral_fee, 2),
            fulfillment_fee=round(self.fba.fulfillment_fee, 2),
            storage_fee=round(storage_fee, 2),
            inbound_fee=round(self.fba.inbound_fee_per_unit, 2),
            total_fba_fee=round(total_fba, 2),
            post_fba_profit=round(post_fba, 2),
            post_fba_margin=round(post_fba_margin, 4),
            ad_spend_per_unit=round(sku.ad_spend_per_unit, 2),
            post_ad_profit=round(post_ad, 2),
            contribution_margin=round(cm, 2),
            contribution_margin_rate=round(cm_rate, 4),
            monthly_units=sku.monthly_units_sold,
            monthly_contribution=round(cm * sku.monthly_units_sold, 2),
            is_profitable=cm > 0,
            primary_cost_driver=primary_driver,
        )

    def portfolio_analysis(self, skus: list) -> dict:
        """组合分析：找盈利/亏损 SKU，计算整体健康度"""
        results = [self.calculate(s) for s in skus]

        profitable = [r for r in results if r.is_profitable]
        losing = [r for r in results if not r.is_profitable]

        total_monthly = sum(r.monthly_contribution for r in results)
        avg_cm_rate = mean([r.contribution_margin_rate for r in results])

        return {
            "results": results,
            "profitable_count": len(profitable),
            "losing_count": len(losing),
            "total_monthly_contribution": round(total_monthly, 2),
            "avg_contribution_margin_rate": round(avg_cm_rate, 4),
            "losing_skus": [r.sku_id for r in losing],
            "top_contributor": max(results, key=lambda r: r.monthly_contribution).sku_id,
        }

    def pricing_scenario(self, sku: SKUCosts, new_price: float,
                         volume_elasticity: float = -1.5) -> dict:
        """定价情景分析：提价 / 降价的 P&L 影响"""
        current = self.calculate(sku)
        price_change_pct = (new_price - sku.selling_price) / sku.selling_price
        new_volume = sku.monthly_units_sold * (1 + volume_elasticity * price_change_pct)

        new_sku = SKUCosts(**{**sku.__dict__,
                              "selling_price": new_price,
                              "monthly_units_sold": max(1, int(new_volume))})
        new_result = self.calculate(new_sku)

        return {
            "current_monthly": current.monthly_contribution,
            "new_monthly": new_result.monthly_contribution,
            "delta_monthly": round(new_result.monthly_contribution - current.monthly_contribution, 2),
            "delta_pct": round((new_result.monthly_contribution - current.monthly_contribution) /
                               max(abs(current.monthly_contribution), 1), 4),
            "new_cm_rate": new_result.contribution_margin_rate,
            "new_volume": new_result.monthly_units,
            "recommendation": "提价有利" if new_result.monthly_contribution > current.monthly_contribution else "维持原价",
        }


def run_sku_pl_demo():
    """演示：母婴产品组合 SKU 级 P&L 分析"""
    print("=" * 65)
    print("SKU-Level P&L Dashboard — 单品利润核算演示")
    print("=" * 65)

    skus = [
        SKUCosts("M5",    "Momcozy M5 Wearable Breast Pump",  89.99, 28.0,  0.06, False, 8.20, 350),
        SKUCosts("S12",   "Momcozy S12 Single Pump",          59.99, 19.5,  0.05, False, 6.80, 180),
        SKUCosts("BAG50", "Storage Bags 50pcs",               12.99, 3.20,  0.04, False, 4.80, 420),
        SKUCosts("UV",    "UV-C Sterilizer",                  49.99, 16.0,  0.07, False, 5.50, 120),
        SKUCosts("BOTT",  "Anti-Colic Bottle Set 3pcs",       24.99, 7.50,  0.08, False, 7.20,  90),
    ]

    calc = SKUPLCalculator()
    portfolio = calc.portfolio_analysis(skus)

    print(f"\n{'SKU':<8} {'售价':>7} {'COGS':>6} {'FBA':>6} {'广告':>6} "
          f"{'CM/件':>7} {'CM率':>6} {'月贡献':>9} {'状态'}")
    print("-" * 75)

    for r in portfolio["results"]:
        status = "✅" if r.is_profitable else "❌亏损"
        print(f"{r.sku_id:<8} ${r.gross_revenue_per_unit:>5.2f} "
              f"${r.cogs:>5.2f} ${r.total_fba_fee:>5.2f} "
              f"${r.ad_spend_per_unit:>5.2f} ${r.contribution_margin:>6.2f} "
              f"{r.contribution_margin_rate:>5.1%} ${r.monthly_contribution:>8,.0f} {status}")

    print(f"\n📊 组合汇总")
    print(f"   盈利 SKU: {portfolio['profitable_count']} / {len(skus)}")
    print(f"   亏损 SKU: {portfolio['losing_count']} — {portfolio['losing_skus']}")
    print(f"   月度总贡献: ${portfolio['total_monthly_contribution']:,.2f}")
    print(f"   平均贡献毛利率: {portfolio['avg_contribution_margin_rate']:.1%}")

    # 定价情景
    m5 = skus[0]
    scenario = calc.pricing_scenario(m5, 94.99, volume_elasticity=-1.5)
    print(f"\n💡 M5 提价 $89.99→$94.99 情景分析:")
    print(f"   当前月贡献: ${scenario['current_monthly']:,.2f}")
    print(f"   提价后月贡献: ${scenario['new_monthly']:,.2f} ({scenario['delta_pct']:+.1%})")
    print(f"   建议: {scenario['recommendation']}")

    # 验证
    results = portfolio["results"]
    assert any(r.sku_id == "BAG50" and not r.is_profitable for r in results), "BAG50 应亏损"
    assert portfolio["top_contributor"] == "M5", "M5 应为最大贡献 SKU"
    assert scenario["delta_monthly"] > 0, "提价应提升月度贡献"

    print("\n[✓] SKU-Level P&L Dashboard 测试通过")
    return portfolio


if __name__ == "__main__":
    run_sku_pl_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-FBA-Fee-Intelligence]]（FBA 费率数据是 SKU P&L 计算的基础输入）
- **前置（prerequisite）**：[[Skill-Ad-Attribution-Modeling]]（广告归因到 SKU 层面是 P&L 中广告费/件计算的前提）
- **延伸（extends）**：[[Skill-PL-Attribution-Analysis]]（SKU P&L 汇总到品牌级 P&L 报告，完成从单品到整体的财务贯通）
- **延伸（extends）**：[[Skill-Forecast-to-PL-Bridge]]（SKU P&L 历史数据 + 需求预测 = 前瞻性财务规划）
- **可组合（combinable）**：[[Skill-Real-Time-Competitive-Repricing]]（组合场景：SKU P&L 设定最低盈亏平衡价 → 重定价系统不得低于该价格）
- **可组合（combinable）**：[[Skill-Inventory-Financing-Optimization]]（组合场景：高贡献毛利 SKU 优先融资备货，低贡献 SKU 降库存）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 发现隐性亏损 SKU：停止亏损广告投放，月节省 $500-3,000
  - 定价情景分析：识别提价空间，月增 GMV ¥5-20 万
  - 资源聚焦高贡献 SKU：FBA 仓储优化节省 $200-800/月
  - **年化综合 ROI**：¥30-100 万

- **实施难度**：⭐☆☆☆☆（公式明确，数据来自 Amazon Seller Central，半天接入）

- **优先级评分**：⭐⭐⭐⭐⭐（图谱 23-运营财务 avg_in 全图最低的根本原因——其他域的 Skill 只有看到"单品赚多少钱"才会主动引用财务 Skill）

- **评估依据**：FBA 费用结构来自 Amazon 官方费率表；贡献毛利分析是 DTC 财务管理标准方法论（参见 arXiv 2211.09612 的利润优化字段验证）
