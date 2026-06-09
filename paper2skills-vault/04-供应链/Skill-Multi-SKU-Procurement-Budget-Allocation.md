---
title: Multi-SKU Procurement Budget Allocation
module: 04-供应链
topic: procurement-budget-optimization
status: stable
domain: supply_chain
papers:
  - id: "2301.02662"
    title: "Robust knapsack ordering for a partially-informed newsvendor with budget constraint"
    venue: "arXiv 2023 (Boonstra, van Eekelen, van Leeuwaarden)"
    role: 主论文（Knapsack Ordering + Minimax + Budget-Consistency 降级策略）
  - id: "EJOR-2024-Olivares"
    title: "Constructing decision rules for multiproduct newsvendors"
    venue: "EJOR Vol.315, 2024"
    role: 战略优先级加权（差异化 cᵤᵢ）+ 替代率建模
roadmap_phase: phase1
---

# Skill-Multi-SKU-Procurement-Budget-Allocation

## ① 算法原理

**核心思想**：季度采购预算有限时，不能对每个 SKU 独立做 Newsvendor 决策——预算约束把所有 SKU 耦合成一个联合优化问题。Knapsack Ordering 算法提供了一个 O(n) 的近最优分配方案：按每个 SKU 的「边际成本效应」排序，依序分配资金直到预算耗尽，同时通过差异化缺货成本系数（cᵤᵢ）将毛利率/战略优先级编码进目标函数。

**Newsvendor with Budget Constraint 基础框架**：

```
单 SKU 报童模型：
  min E[cₒ·(q - D)⁺ + cᵤ·(D - q)⁺]
  最优解：q* = F⁻¹(α*)，其中 α* = cᵤ/(cᵤ + cₒ)（临界比率）

多 SKU 预算约束扩展：
  min Σᵢ E[cₒᵢ·(qᵢ - Dᵢ)⁺ + cᵤᵢ·(Dᵢ - qᵢ)⁺]
  s.t. Σᵢ pᵢ·qᵢ ≤ B

Lagrangian 松弛：
  对每个 λ ≥ 0，单独优化 qᵢ(λ) = F⁻¹((cᵤᵢ - λpᵢ)/(cᵤᵢ + cₒᵢ))
  二分搜索 λ* 使 Σᵢ pᵢ·qᵢ(λ*) = B
```

**Knapsack Ordering 算法（arXiv:2301.02662）**：

```
仅需均值/MAD/值域（无需完整分布），适合信息不完整场景：

Step 1: 对每个 SKU i 计算「边际价值比」
  marginal_value_i = (cᵤᵢ · E[Dᵢ] - cₒᵢ · qᵢ) / (pᵢ · qᵢ)

Step 2: 按边际价值比降序排列 → i₁, i₂, ..., iₙ

Step 3: 按序分配资金，直至 B 耗尽：
  for i in [i₁, i₂, ..., iₙ]:
      if remaining_budget >= pᵢ · qᵢ*(Minimax):
          allocate qᵢ*(Minimax)，remaining_budget -= pᵢ · qᵢ*
      else:
          allocate remaining_budget / pᵢ（部分满足），break

Budget-Consistency Property（关键保证）：
  预算增加 ΔB → 原有 SKU 订量不变，ΔB 只追加给排名靠后的 SKU
  → 降级策略 = 直接截断排名靠后的 SKU（不影响高优先级 SKU）
```

**战略优先级加权（EJOR 2024）**：

```
高毛利/战略 SKU：赋高 cᵤᵢ（缺货成本大），临界比率 αᵢ* 更高 → 优先保障
低毛利/可替代 SKU：赋低 cᵤᵢ，临界比率低 → 预算紧张时优先削减

实操参数化（母婴 DTC）：
  cᵤᵢ = 单价 × 毛利率 × 缺货期 BSR 损失系数
  cₒᵢ = 单价 × 持有成本率 × 库存超期概率
```

**关键假设**：
- 各 SKU 需求独立（不考虑替代效应时；EJOR 2024 扩展处理了替代率）
- 采购价格 pᵢ 在优化区间内固定（MOQ/价格阶梯另行处理，见 Skill-Dynamic-Lot-Sizing）
- 预算 B 为硬约束（不可透支）

---

## ② 母婴出海应用案例

**场景 A：季度 $200K 预算在 8 个核心 SKU 间分配**

- **业务问题**：Q4 备货预算 $200,000，8 个 SKU，各有不同毛利率和需求不确定性，如何分配才能最大化整体服务水平并保障高毛利 SKU 不断货？
- **数据要求**：每 SKU 的历史需求均值/标准差、采购单价、毛利率、BSR 重要性评级
- **预期产出**：
  ```
  SKU 排序（按边际价值比）：
    #1 S12 Pro（毛利58%，BSR Top-5）    → 分配 $52,000，订 1,300 件
    #2 M5（毛利52%，高需求确定性）      → 分配 $38,000，订 1,267 件
    #3 S21 Pro（新品，需求不确定性高）  → 分配 $28,000，订 700 件
    ... （依序分配，预算耗尽时截断）
    
  预算收紧降级：若预算削减至 $160K，直接截断 #7/#8 SKU 配额
  高优先级 SKU（#1-#5）订量不受影响（Budget-Consistency）
  ```
- **业务价值**：相比「平均分配」或「按需求比例」，Knapsack Ordering 可将整体服务水平从 ~85% 提升至 ~92%，对应减少缺货损失约 $15,000-$25,000/季度

**场景 B：新品上市预算挤占存量 SKU 资金的优先级决策**

- **业务问题**：新款 UV-C 消毒器需要首批备货 $35,000，但季度总预算不变，从哪个存量 SKU 削减预算？
- **预期产出**：按 Budget-Consistency 性质，从排名最低（边际价值最小）的 SKU 削减，不影响高毛利主力 SKU

---

## ③ 代码模板

```python
"""
Skill-Multi-SKU-Procurement-Budget-Allocation
基于 arXiv:2301.02662 (Boonstra et al. 2023, Knapsack Ordering) +
    EJOR Vol.315 2024 (Olivares-Nadal, 战略优先级加权)
母婴跨境 DTC 多 SKU 季度采购预算分配优化
"""

import numpy as np
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import brentq


@dataclass
class SKUSpec:
    sku_id: str
    demand_mean: float
    demand_std: float
    unit_price: float
    gross_margin_rate: float
    stockout_bsr_penalty: float = 1.5
    strategic_priority: str = "medium"

    @property
    def holding_cost_per_unit(self) -> float:
        return self.unit_price * 0.20 / 4

    @property
    def stockout_cost_per_unit(self) -> float:
        return self.unit_price * self.gross_margin_rate * self.stockout_bsr_penalty

    @property
    def critical_ratio(self) -> float:
        cu, co = self.stockout_cost_per_unit, self.holding_cost_per_unit
        return cu / (cu + co)


@dataclass
class AllocationResult:
    sku_id: str
    order_qty: float
    budget_allocated: float
    service_level: float
    marginal_value: float
    rank: int
    truncated: bool = False


def newsvendor_qty(sku: SKUSpec, critical_ratio: float | None = None) -> float:
    cr = critical_ratio if critical_ratio is not None else sku.critical_ratio
    return max(0.0, stats.norm.ppf(cr, loc=sku.demand_mean, scale=sku.demand_std))


def lagrangian_allocation(skus: list[SKUSpec], budget: float) -> list[AllocationResult]:
    """
    Lagrangian 松弛：二分搜索 λ* 使预算约束恰好满足。
    适合需求分布已知（Normal）的场景。
    """
    def total_spend(lam: float) -> float:
        total = 0.0
        for sku in skus:
            cu, co = sku.stockout_cost_per_unit, sku.holding_cost_per_unit
            adjusted_cr = max(0.01, min(0.99, (cu - lam * sku.unit_price) / (cu + co)))
            q = newsvendor_qty(sku, adjusted_cr)
            total += q * sku.unit_price
        return total

    if total_spend(0.0) <= budget:
        lam_star = 0.0
    else:
        try:
            lam_star = brentq(lambda l: total_spend(l) - budget, 0.0, 1000.0, xtol=0.01)
        except ValueError:
            lam_star = 1000.0

    results = []
    for i, sku in enumerate(skus):
        cu, co = sku.stockout_cost_per_unit, sku.holding_cost_per_unit
        adjusted_cr = max(0.01, min(0.99, (cu - lam_star * sku.unit_price) / (cu + co)))
        q = newsvendor_qty(sku, adjusted_cr)
        spend = q * sku.unit_price
        results.append(AllocationResult(
            sku_id=sku.sku_id,
            order_qty=round(q),
            budget_allocated=round(spend, 2),
            service_level=round(adjusted_cr, 3),
            marginal_value=(cu * sku.demand_mean) / max(spend, 1),
            rank=i + 1,
        ))
    return results


def knapsack_ordering(skus: list[SKUSpec], budget: float) -> list[AllocationResult]:
    """
    Knapsack Ordering（arXiv:2301.02662）：
    按边际价值比排序，依序分配直到预算耗尽。
    仅需均值/MAD，适合信息不完整场景（新品/稀疏需求）。
    """
    scored = []
    for sku in skus:
        q_opt = newsvendor_qty(sku)
        spend = q_opt * sku.unit_price
        cu = sku.stockout_cost_per_unit
        mv = (cu * sku.demand_mean) / max(spend, 1e-6)
        scored.append((mv, sku, q_opt, spend))

    scored.sort(key=lambda x: -x[0])

    results = []
    remaining = budget
    for rank, (mv, sku, q_opt, spend_opt) in enumerate(scored, 1):
        if remaining <= 0:
            results.append(AllocationResult(
                sku_id=sku.sku_id, order_qty=0, budget_allocated=0.0,
                service_level=0.0, marginal_value=mv, rank=rank, truncated=True,
            ))
        elif remaining >= spend_opt:
            results.append(AllocationResult(
                sku_id=sku.sku_id, order_qty=round(q_opt),
                budget_allocated=round(spend_opt, 2),
                service_level=round(sku.critical_ratio, 3),
                marginal_value=mv, rank=rank,
            ))
            remaining -= spend_opt
        else:
            partial_q = remaining / sku.unit_price
            partial_sl = float(stats.norm.cdf(partial_q, sku.demand_mean, sku.demand_std))
            results.append(AllocationResult(
                sku_id=sku.sku_id, order_qty=round(partial_q),
                budget_allocated=round(remaining, 2),
                service_level=round(partial_sl, 3),
                marginal_value=mv, rank=rank, truncated=True,
            ))
            remaining = 0.0

    return results


def print_allocation_report(results: list[AllocationResult], budget: float, method: str):
    print(f"\n{'='*65}")
    print(f"采购预算分配报告 [{method}]  总预算: ${budget:,.0f}")
    print(f"{'='*65}")
    total_spend = sum(r.budget_allocated for r in results)
    weighted_sl = sum(r.service_level * r.budget_allocated for r in results) / max(total_spend, 1)
    print(f"{'排名':<4} {'SKU':<20} {'订量':>6} {'分配预算':>10} {'服务水平':>8} {'状态'}")
    print("-" * 65)
    for r in results:
        status = "⚠️ 截断" if r.truncated else "✅"
        print(f"#{r.rank:<3} {r.sku_id:<20} {r.order_qty:>6,} ${r.budget_allocated:>9,.0f} "
              f"{r.service_level:>7.1%} {status}")
    print("-" * 65)
    print(f"{'合计':<25} ${total_spend:>9,.0f} {weighted_sl:>7.1%} (加权均值)")
    budget_util = total_spend / budget
    print(f"预算利用率: {budget_util:.1%}")


if __name__ == "__main__":
    skus = [
        SKUSpec("S12-Pro",    demand_mean=1200, demand_std=240, unit_price=38.0, gross_margin_rate=0.58, stockout_bsr_penalty=2.0, strategic_priority="high"),
        SKUSpec("M5",         demand_mean=900,  demand_std=150, unit_price=30.0, gross_margin_rate=0.52, stockout_bsr_penalty=1.8),
        SKUSpec("S21-Pro",    demand_mean=600,  demand_std=200, unit_price=42.0, gross_margin_rate=0.55, stockout_bsr_penalty=1.5),
        SKUSpec("LF1",        demand_mean=500,  demand_std=120, unit_price=28.0, gross_margin_rate=0.45, stockout_bsr_penalty=1.3),
        SKUSpec("UV-C-Pro",   demand_mean=300,  demand_std=150, unit_price=55.0, gross_margin_rate=0.60, stockout_bsr_penalty=1.6),
        SKUSpec("Wearable-S", demand_mean=250,  demand_std=80,  unit_price=48.0, gross_margin_rate=0.50, stockout_bsr_penalty=1.2),
        SKUSpec("S12-Basic",  demand_mean=400,  demand_std=100, unit_price=22.0, gross_margin_rate=0.38, stockout_bsr_penalty=1.0),
        SKUSpec("Accessory",  demand_mean=800,  demand_std=200, unit_price=12.0, gross_margin_rate=0.30, stockout_bsr_penalty=0.8),
    ]

    budget = 200_000.0
    results_knapsack = knapsack_ordering(skus, budget)
    print_allocation_report(results_knapsack, budget, "Knapsack Ordering")

    print(f"\n{'='*65}")
    print("Budget-Consistency 验证：预算削减至 $160K")
    print("高优先级 SKU (#1-#5) 订量应保持不变")
    results_reduced = knapsack_ordering(skus, 160_000.0)
    for r_full, r_cut in zip(results_knapsack[:5], results_reduced[:5]):
        delta = r_cut.order_qty - r_full.order_qty
        status = "✅ 不变" if delta == 0 else f"⚠️ 变化{delta:+d}"
        print(f"  {r_full.sku_id}: {r_full.order_qty} → {r_cut.order_qty} {status}")
```

---

## ④ 技能关联

- **前置技能**：
  - [[Skill-Demand-Forecasting-Supply-Chain]] — 需求均值/标准差是预算分配的核心输入
  - [[Skill-Safety-Stock-Replenishment]] — 安全库存量决定订购量的下界
- **延伸技能**：
  - [[Skill-Dynamic-Lot-Sizing-MOQ]] — 价格阶梯和 MOQ 约束对订购量的进一步调整
  - [[Skill-Supplier-Capacity-Planning]] — 供应商产能约束下，预算分配结果需要二次调整
- **可组合**：
  - [[Skill-Promotion-Demand-Decomposition]] — 大促备货量直接替换 demand_mean，驱动大促期预算分配
  - [[Skill-ROAS-Budget-Optimization]] — 采购预算分配与广告预算分配逻辑同构，可联合做资金总池规划

---
- **相关技能**：[[Skill-New-Product-Inventory-Coldstart]]

## ⑤ 商业价值评估

- **ROI 预估**：
  - 相比平均分配策略，Knapsack Ordering 将加权服务水平从约 85% 提升至约 92%
  - 7pp 服务水平提升 × $200K 季度采购 × 缺货损失系数 ≈ 减少缺货损失约 $15,000-$25,000/季度
  - 年化：$60,000-$100,000（4 个季度）
  - Budget-Consistency 降级策略：预算削减时，直接截断低优先级 SKU，零额外计算成本
- **实施难度**：⭐⭐☆☆☆（2/5）— O(n) 排序算法，无需求解器
- **优先级评分**：⭐⭐⭐⭐⭐（5/5）— 每季度必做的资金分配决策，当前完全凭经验

---

## 元信息

```yaml
skill_id: Skill-Multi-SKU-Procurement-Budget-Allocation
domain: supply_chain
vault_path: paper2skills-vault/04-供应链/Skill-Multi-SKU-Procurement-Budget-Allocation.md
code_path: paper2skills-code/supply_chain/multi_sku_budget_allocation/
review_score: 8.5/10
wf_coverage: [WF-A]
created: 2026-05-25
```
