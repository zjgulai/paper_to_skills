---
title: Skill-Inventory-Carrying-Cost-Model — 库存持有成本模型
doc_type: knowledge
module: 23-运营财务
topic: inventory-carrying-cost-model
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Inventory-Carrying-Cost-Model

## ① 算法原理（≤300字）

库存持有成本（Inventory Carrying Cost）是持有库存所产生的全部隐性和显性成本总和，通常占库存货值的 25-35%/年，是卖家最容易忽视的利润黑洞。

**全口径成本组成**：
1. **资金成本（Capital Cost）**：库存货值 × 资金年化成本率（融资成本或机会成本，通常 15-20%）
2. **仓储成本（Storage Cost）**：FBA 月均库存费 + 海外仓费 + 国内仓费
3. **长期仓储罚款（LTSF）**：超龄库存的 Amazon 惩罚性仓储费
4. **保险成本（Insurance）**：货物保险，约货值的 0.5-1%/年
5. **货物折损风险（Shrinkage）**：破损、丢失、过期，约 1-2%/年
6. **机会成本（Opportunity Cost）**：资金用于库存而非更高收益投资的损失

**持有成本率计算**：
```
年化持有成本率 = (资金成本 + 仓储成本 + LTSF + 保险 + 折损 + 机会成本) / 平均库存货值
```

**最优订货量（EOQ）**：平衡订货成本与持有成本的数学模型，找到使总成本最小化的订货批量。

## ② 母婴出海应用案例

**场景**：母婴品牌月销 3,000 件吸奶器，每件货值 $25，当前平均库存 4,500 件（1.5 个月库存），总货值 $112,500。

持有成本全口径核算：
- 资金成本（年化 18%）：$112,500 × 18% = $20,250/年
- FBA 月仓储费：$1,800/月 × 12 = $21,600/年
- LTSF（180 天+库存占比 15%）：$2,700/年
- 折损保险估算：$1,125/年
- **总持有成本：$45,675/年（货值的 40.6%）**

通过 EOQ 模型，将安全库存降至 0.8 个月，持有成本降至 $24,000/年，**年化节省 $21,675（约 15 万元）**。

## ③ 代码模板

```python
import numpy as np
import pandas as pd

# 库存持有成本模型

def compute_carrying_cost(
    avg_inventory_value: float,       # 平均库存货值（USD）
    capital_cost_rate: float = 0.18,  # 资金年化成本率
    storage_cost_annual: float = 0.0, # 年仓储费（USD）
    ltsf_annual: float = 0.0,         # 年LTSF罚款（USD）
    insurance_rate: float = 0.007,    # 保险率
    shrinkage_rate: float = 0.015,    # 折损率
) -> dict:
    """计算全口径年化库存持有成本"""
    capital_cost = avg_inventory_value * capital_cost_rate
    insurance = avg_inventory_value * insurance_rate
    shrinkage = avg_inventory_value * shrinkage_rate

    total = capital_cost + storage_cost_annual + ltsf_annual + insurance + shrinkage
    carrying_rate = total / avg_inventory_value

    return {
        '资金成本(USD)': round(capital_cost, 0),
        '仓储成本(USD)': round(storage_cost_annual, 0),
        'LTSF罚款(USD)': round(ltsf_annual, 0),
        '保险成本(USD)': round(insurance, 0),
        '折损成本(USD)': round(shrinkage, 0),
        '总持有成本(USD)': round(total, 0),
        '年化持有成本率': f'{carrying_rate:.1%}',
    }


def eoq_model(
    annual_demand: float,      # 年需求量（件）
    unit_cost: float,          # 单件成本（USD）
    order_cost: float,         # 每次订货成本（USD）
    carrying_rate: float,      # 年化持有成本率
) -> dict:
    """经济订货量（EOQ）模型"""
    eoq = np.sqrt(2 * annual_demand * order_cost / (unit_cost * carrying_rate))
    num_orders = annual_demand / eoq
    avg_inventory = eoq / 2
    total_ordering_cost = num_orders * order_cost
    total_carrying_cost = avg_inventory * unit_cost * carrying_rate
    total_cost = total_ordering_cost + total_carrying_cost

    return {
        'EOQ（件）': round(eoq, 0),
        '年订货次数': round(num_orders, 1),
        '平均库存（件）': round(avg_inventory, 0),
        '年订货成本(USD)': round(total_ordering_cost, 0),
        '年持有成本(USD)': round(total_carrying_cost, 0),
        '年总成本(USD)': round(total_cost, 0),
    }


def sensitivity_analysis_carrying(
    avg_inventory_value: float,
    storage_cost_annual: float,
    capital_rates: list = None
) -> pd.DataFrame:
    """资金成本率敏感性分析"""
    if capital_rates is None:
        capital_rates = [0.10, 0.15, 0.18, 0.20, 0.25]

    rows = []
    for rate in capital_rates:
        cc = compute_carrying_cost(avg_inventory_value, rate, storage_cost_annual)
        rows.append({'资金成本率': f'{rate:.0%}', **cc})
    return pd.DataFrame(rows)


# ── 测试 ──
if __name__ == '__main__':
    # 吸奶器场景
    avg_value = 112500  # 4500件 × $25

    print("=== 全口径库存持有成本 ===")
    cc = compute_carrying_cost(
        avg_inventory_value=avg_value,
        capital_cost_rate=0.18,
        storage_cost_annual=21600,
        ltsf_annual=2700,
    )
    for k, v in cc.items():
        print(f"  {k}: {v}")

    print("\n=== EOQ最优订货量 ===")
    eoq = eoq_model(
        annual_demand=36000,  # 月3000件 × 12
        unit_cost=25,
        order_cost=500,
        carrying_rate=0.40,
    )
    for k, v in eoq.items():
        print(f"  {k}: {v}")

    print("\n=== 资金成本率敏感性分析 ===")
    sa = sensitivity_analysis_carrying(avg_value, 21600)
    print(sa[['资金成本率', '总持有成本(USD)', '年化持有成本率']].to_string(index=False))
    print(f"\n[✓] 库存持有成本模型测试通过")
```


## ④ 技能关联

- 前置技能：[[Skill-Inventory-Health-Aging-Attribution]]
- 前置技能：[[Skill-FBA-Cost-Forecast-Adjustment]]
- 延伸技能：[[Skill-GMROI-Inventory-Investment-Efficiency]]
- 延伸技能：[[Skill-Supply-Chain-Total-Cost-TCO-Model]]
- 可组合：[[Skill-Markdown-Optimization]]
- 可组合：[[Skill-Safety-Stock-Replenishment]]

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| ROI | 识别隐性持有成本，年化节省 10-50 万元 |
| 实施难度 | ⭐⭐（数据可从 Amazon 账单获取） |
| 优先级 | ⭐⭐⭐⭐（库存积压严重时立即触发） |
| 数据要求 | 平均库存货值 + FBA 月度仓储报告 + 资金成本率 |
| 典型收益 | 识别 40% 年化持有成本率，通过 EOQ 优化降至 20-25% |
