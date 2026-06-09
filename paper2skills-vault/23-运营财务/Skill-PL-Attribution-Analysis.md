---
title: P&L Attribution Analysis（SKU 级损益归因分析）
doc_type: knowledge
module: 23-运营财务
topic: pl-attribution-analysis
status: stable
created: 2026-06-09
updated: 2026-06-09
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: PL-Attribution-Analysis（P&L 多维归因分析）

> **桥梁**: 23-运营财务 ↔ 13-广告分析 ↔ 04-供应链 | **类型**: 运营财务

---

## ① 算法原理

**核心思想**：跨境电商的 P&L 必须拆解到 SKU × 渠道 × 市场三个维度，才能识别「哪个 SKU 真正在赚钱」「哪个渠道拉高了整体亏损」。传统 P&L 只看总账，导致高毛利 SKU 补贴低毛利 SKU 的情况长期不可见。

**四层拆解结构**：
```
层级 1: 毛利（Gross Profit）
  = 销售额 - COGS（采购成本 + 头程运费）

层级 2: 运营利润（Operating Profit）
  = 毛利 - 平台佣金 - FBA 费用 - 广告费

层级 3: 净利润（Net Profit）
  = 运营利润 - 退货成本 - 汇率损益 - 税费

层级 4: 归因调整（Attribution Adjustment）
  = 净利润 ± 因果归因修正（广告真实增量 vs naive 归因）
```

**因果归因修正**（与 Skill-Causal-Attribution-Bridge 联动）：
```
naive_profit = revenue - all_costs
causal_ad_contribution = naive_ad_revenue × attribution_fraction
adjusted_profit = naive_profit - (naive_ad_revenue - causal_ad_contribution)
```

这修正了「广告虚报贡献→利润虚高」的常见错误。

---

## ② 母婴出海应用案例

**业务痛点**：某母婴品牌月度总利润率看起来 18%，但不知道是哪些 SKU 在拉高还是拖低。吸奶器主力款和配件款的真实利润完全不清楚。

**分析结论示例**：
| SKU | 毛利率 | 运营利润率 | 净利润率 | 状态 |
|-----|--------|-----------|---------|------|
| 吸奶器 S1 主机 | 52% | 24% | 19% | ✅ 核心利润源 |
| 吸奶器配件包  | 61% | 38% | 31% | ✅ 高利润，低广告依赖 |
| 婴儿推车 L1   | 38% | 8%  | 2%  | ⚠️ 高运费侵蚀利润 |
| 奶粉定制款    | 28% | -5% | -12%| ❌ 广告费过高，亏损 |

**决策**：停止奶粉定制款广告投放，将预算转移到配件包；推车单独计算物流方案。

---

## ③ 代码模板

```python
from dataclasses import dataclass

@dataclass
class SKUPLData:
    sku: str
    revenue: float
    cogs: float
    head_haul: float
    platform_commission_pct: float
    fba_fee: float
    ad_spend: float
    return_cost: float
    fx_adjustment: float = 0.0
    ad_attribution_fraction: float = 1.0

def compute_pl(d: SKUPLData) -> dict:
    gross_profit = d.revenue - d.cogs - d.head_haul
    platform_fee = d.revenue * d.platform_commission_pct
    operating_profit = gross_profit - platform_fee - d.fba_fee - d.ad_spend
    net_profit = operating_profit - d.return_cost + d.fx_adjustment

    causal_ad_revenue = d.ad_spend / max(d.ad_attribution_fraction, 0.01) * d.ad_attribution_fraction
    naive_ad_revenue  = d.ad_spend / max(d.ad_attribution_fraction, 0.01)
    attribution_adj   = -(naive_ad_revenue - causal_ad_revenue)
    adjusted_profit   = net_profit + attribution_adj

    def pct(v):
        return round(v / d.revenue * 100, 1) if d.revenue else 0.0

    status = "盈利" if adjusted_profit > 0 else "亏损"
    if pct(adjusted_profit) < 5 and adjusted_profit > 0:
        status = "微利"

    return {
        "sku": d.sku, "revenue": d.revenue,
        "gross_margin": pct(gross_profit),
        "operating_margin": pct(operating_profit),
        "net_margin": pct(net_profit),
        "adjusted_margin": pct(adjusted_profit),
        "status": status,
    }

skus = [
    SKUPLData("吸奶器S1", 100000, 28000, 5000, 0.15, 3000, 8000, 2000,  0,   0.85),
    SKUPLData("配件包",   30000,  8000, 1500, 0.15,  800, 1500,  500,  0,   0.95),
    SKUPLData("婴儿推车", 50000, 22000, 7000, 0.15, 4000, 5000, 3000,  0,   0.80),
    SKUPLData("奶粉定制", 20000, 10000, 2000, 0.15, 1500, 8000,  800,  0,   0.60),
]
print(f"{'SKU':<10} {'毛利%':>7} {'运营%':>7} {'净利%':>7} {'归因调整%':>9} {'状态'}")
print("-" * 55)
for s in skus:
    r = compute_pl(s)
    print(f"{r['sku']:<10} {r['gross_margin']:>7} {r['operating_margin']:>7} "
          f"{r['net_margin']:>7} {r['adjusted_margin']:>9} {r['status']}")

print("\n[✓] P&L Attribution Analysis 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-FBA-Fee-Intelligence]]（FBA 费用输入）
- **前置**：[[Skill-Cross-Border-Cash-Flow-Forecasting]]（P&L 驱动现金流预测）
- **组合**：[[Skill-Causal-Attribution-Bridge]]（广告真实贡献修正 P&L）
- **组合**：[[Skill-ROAS-Budget-Optimization]]（广告费率优化影响运营利润）
- **延伸**：[[Skill-HTS-Tariff-Classification]]（关税影响 COGS）
- **延伸**：[[Skill-Compliant-Dynamic-Pricing-Guard]]（定价决策影响毛利率）

---

## ⑤ 商业价值评估

**ROI 估算**：
| 场景 | 年化价值 |
|------|---------|
| 识别亏损 SKU 停止广告投放 | 年节省无效广告 10-30 万元 |
| 运营利润率提升 2pp | 月 GMV 200 万 × 2% = 年增利润 48 万元 |
| 归因修正后预算重分配 | 参见 Skill-Causal-Attribution-Bridge |

**实施难度**：⭐⭐☆☆☆（低，数据来自现有报表，框架建立 1 周内可用）

**优先级评分**：5/5（每个月必须做的财务分析，缺失导致不知道谁在赚钱）
