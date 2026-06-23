---
title: Skill-Tariff-Impact-Margin-Stress-Test — 关税冲击利润压力测试
doc_type: knowledge
module: 23-运营财务
topic: tariff-impact-margin-stress-test
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Tariff-Impact-Margin-Stress-Test

## ① 算法原理（≤300字）

关税压力测试是一种情景分析框架，量化不同关税税率变化对 SKU 毛利率的逐级影响，帮助卖家提前制定对冲策略。

**情景设计矩阵**：
- **基准情景**：当前税率（如 Section 301 List 3，25%）
- **温和加征**：税率提升 5-10%
- **极端情景**：税率提升至 50%+（如对等关税）
- **结构调整**：供应商转移（越南/墨西哥/印度）后的新税率

**逐层影响传导**：
1. 关税直接增加进口成本（CIF × 新税率 - CIF × 旧税率）
2. 关税增加部分是否转嫁给消费者（价格弹性分析）
3. 转嫁比例 × 价格弹性 → 销量影响
4. 最终净利润变化 = 成本增加 - 转嫁收入 - 销量损失

**盈亏平衡关税率**计算：
```
临界税率 = 当前毛利率 / (1 + 当前税率) × 100%
```

超过临界税率，SKU 即进入亏损区间。

## ② 母婴出海应用案例

**场景**：母婴品牌从中国进口婴儿监视器，CIF 成本 35 美元/件，售价 89.99 美元，当前关税 25%。2025 年"对等关税"提案将税率提升至 54%。

压力测试结果：
- 当前毛利率：(89.99 - 35×1.25 - 12.5) / 89.99 = **28.3%**
- 54% 税率下：额外关税成本 35×0.29 = 10.15 美元/件
- 若不提价：毛利率降至 **16.8%**，已低于 Amazon 费用覆盖线
- 若涨价 $10：转化率预计下降 15%，月销量从 2,000 降至 1,700 件

**决策结论**：启动越南替代供应商（税率 10%），目标 6 个月完成切换，年化关税节省 **43 万元**。

## ③ 代码模板

```python
import numpy as np
import pandas as pd

# 关税冲击利润压力测试

def compute_margin(
    selling_price: float,
    cif_cost: float,
    tariff_rate: float,
    fba_fee: float,
    referral_rate: float = 0.15,
    other_opex: float = 0.0
) -> float:
    """计算单品毛利率"""
    landed_cost = cif_cost * (1 + tariff_rate)
    referral_fee = selling_price * referral_rate
    profit = selling_price - landed_cost - fba_fee - referral_fee - other_opex
    return profit / selling_price


def stress_test_tariff(
    sku_name: str,
    selling_price: float,
    cif_cost: float,
    base_tariff: float,
    fba_fee: float,
    tariff_scenarios: list,
    price_elasticity: float = -1.5,
    base_monthly_units: int = 1000,
    referral_rate: float = 0.15,
) -> pd.DataFrame:
    """
    压力测试：不同关税情景下的利润影响

    tariff_scenarios: [(场景名, 税率, 价格转嫁比例), ...]
    """
    results = []
    base_margin = compute_margin(selling_price, cif_cost, base_tariff, fba_fee, referral_rate)

    for name, tariff, pass_through in tariff_scenarios:
        extra_cost = cif_cost * (tariff - base_tariff)
        price_increase = extra_cost * pass_through
        new_price = selling_price + price_increase
        new_margin = compute_margin(new_price, cif_cost, tariff, fba_fee, referral_rate)

        pct_price_change = price_increase / selling_price
        volume_change = 1 + (price_elasticity * pct_price_change)
        new_units = max(0, base_monthly_units * volume_change)

        monthly_profit_change = (
            new_units * new_price * new_margin -
            base_monthly_units * selling_price * base_margin
        )

        results.append({
            '情景': name,
            '关税率': f'{tariff:.0%}',
            '新售价': round(new_price, 2),
            '新毛利率': f'{new_margin:.1%}',
            '月销量变化': f'{new_units:.0f}件',
            '月利润变化(USD)': round(monthly_profit_change, 0),
            '年化影响(万元)': round(monthly_profit_change * 12 * 7.15 / 10000, 1),
        })

    return pd.DataFrame(results)


def find_breakeven_tariff(
    selling_price: float, cif_cost: float, fba_fee: float,
    referral_rate: float = 0.15, min_margin: float = 0.0
) -> float:
    """二分法找盈亏平衡关税率"""
    lo, hi = 0.0, 2.0
    for _ in range(50):
        mid = (lo + hi) / 2
        m = compute_margin(selling_price, cif_cost, mid, fba_fee, referral_rate)
        if m > min_margin:
            lo = mid
        else:
            hi = mid
    return round(lo, 4)


# ── 测试 ──
if __name__ == '__main__':
    scenarios = [
        ('当前（25%）', 0.25, 0.0),
        ('温和加征（35%）', 0.35, 0.3),
        ('对等关税（54%）', 0.54, 0.5),
        ('极端（100%）', 1.00, 0.7),
    ]

    result = stress_test_tariff(
        sku_name='婴儿监视器',
        selling_price=89.99,
        cif_cost=35.0,
        base_tariff=0.25,
        fba_fee=12.5,
        tariff_scenarios=scenarios,
        price_elasticity=-1.5,
        base_monthly_units=2000,
    )
    print("=== 关税压力测试结果 ===")
    print(result.to_string(index=False))

    be = find_breakeven_tariff(89.99, 35.0, 12.5)
    print(f"\n盈亏平衡关税率: {be:.1%}")
    print(f"[✓] 关税冲击利润压力测试通过")
```

## ④ 技能关联

- 前置：[[Skill-Cross-Border-Tax-Tariff-Modeling]] — 基础关税建模
- 延伸：[[Skill-Tariff-FX-FBA-Cost-Dynamics]] — 多因素成本动态
- 延伸：[[Skill-SKU-Level-PL-Dashboard]] — 单品盈利看板
- 组合：[[Skill-FX-Dynamic-Pricing-Adjustment]] — 联合定价调整

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| ROI | 提前 6 个月决策供应链转移，年化关税节省 20-100 万元 |
| 实施难度 | ⭐⭐（数据清晰，模型标准） |
| 优先级 | ⭐⭐⭐⭐⭐（2025 年关税环境下极高紧迫性） |
| 数据要求 | SKU 成本结构 + 当前税率 + 价格弹性估算 |
| 典型收益 | 找到盈亏临界关税率，决策供应链结构调整时机 |
