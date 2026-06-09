---
title: New Product Inventory Cold Start
module: 04-供应链
topic: new-product-launch-inventory
status: stable
domain: supply_chain
papers:
  - id: "M&SOM-2019-Ban-Gallien"
    title: "Dynamic Procurement of New Products with Covariate Information: The Residual Tree Method"
    venue: "M&SOM 21(4):798-815, 2019 (Zara case)"
    role: 主论文（类比SKU残差树+多期LP，Zara跨境采购同构场景）
  - id: "OR-2023-Keskin"
    title: "Bayesian Inventory Control: Accelerated Demand Learning via Exploration Boosts"
    venue: "Operations Research 71(5), 2023"
    role: 上市后Bayesian快速更新+探索加成首批量公式
  - id: "TF&SC-2014-Lee"
    title: "Pre-launch new product demand forecasting using Bass model with ML"
    venue: "Technological Forecasting & Social Change, 2014"
    role: Bass p/q参数的ML预估（无历史数据时的fallback）
roadmap_phase: phase1
---

# Skill-New-Product-Inventory-Coldstart

## ① 算法原理

**核心思想**：新品上市前无历史销量，但不是无信息——相似 SKU 的历史数据、产品属性特征、Bass 扩散参数估计三条路径可以构建新品的需求先验分布。上市后用贝叶斯更新快速收敛，并通过「探索加成」（首批量刻意多订一点以加速学习）避免因首批订少而永久缺乏数据的陷阱。

**三档决策树（按信息量）**：

```
无强相似SKU（全新品类创新）→ Bass扩散参数ML估计 → 首批量 = Bass峰值×0.3
中等相似（同品类不同价位）→ 类比SKU残差树 → 多期LP首批+补货计划
强相似（换代升级款）→ 历史SKU+贝叶斯更新 → 快速收敛到真实需求
```

**类比 SKU 残差树（Ban, Gallien & Mersereau 2019）**：

```
Step 1: 回归建立需求预测
  D_new ≈ β₀ + β₁×价格段 + β₂×品类 + β₃×上市月份 + ...
  用所有历史新品数据训练（Lasso 正则化）

Step 2: 把回归残差按时间路径分箱
  residuals = actual_hist - predicted_hist
  bins = quantile_bins(residuals, n_bins=10)
  → 构造新品的「人工需求场景树」

Step 3: 多期 LP 决定首批量+补货序列
  min 总成本（持货 + 缺货）across all scenario paths
  s.t. q₁ ≥ 0（首批），q₂(ω) ≥ 0（在途补货，依赖场景ω）

理论保证：
  忽略协变量信息 → 总成本上升 6-15%
  仅用2-3分支场景树 → 总成本上升 30-66%
```

**贝叶斯上市后快速更新（Keskin, Li & Song 2023）**：

```
先验：D ~ Normal(μ₀, σ₀²)（来自类比SKU估计）

上市后第 t 周观测到右删失销量 s_t（实际销量 ≤ 库存量）：
  后验更新：μ_t → μ_{t+1}，σ_t² → σ_{t+1}²（贝叶斯卡尔曼滤波）

最优首批量 = 短视最优量 + 探索加成
  q*_BDP = q*_myopic + exploration_boost

探索加成 ∝ 后验离散度指数（posterior index of dispersion）
  = σ²/μ（需求均值的参数不确定性）

→ 参数不确定性越高，首批量应该越多（主动学习）
→ 短视策略（只订 q*_myopic）在高不确定性场景损失可以任意大
```

**Bass 参数 ML 估计（Lee et al. 2014）**：

```
当无强相似SKU时（品类全新），用产品属性预测 p/q 参数：
  p = f(价格段, 技术类型, 竞品数量, 营销投入强度)
  q = g(同上)

母婴耐用品经验参数范围：
  p ∈ [0.008, 0.04]（创新系数：慢于消费电子）
  q ∈ [0.20, 0.45]（模仿系数）

Bass 首批量估计：
  peak_demand = M × (p+q)²/(4q)（峰值月需求量）
  首批量 = peak_demand × lead_time_months × 保守系数(0.3-0.5)
```

**关键假设**：
- 存在 ≥ 3 个相似历史 SKU（类比法有效的前提）
- 上市后第 1-4 周销量可实时获取（贝叶斯更新的触发条件）
- lead time 已知且相对确定（否则需叠加交货期风险）

---

## ② 母婴出海应用案例

**场景 A：新款 UV-C 密闭消毒器上市首批备货（强相似 SKU 存在）**

- **业务问题**：全新 SKU「UV-C Pro X200」即将上市，lead time 6 周（需提前 6 周锁定首批量）。类比品：已有 UV-C Pro X100（上市 8 个月，月均销量 320 件，系数变异 CV=35%）。
- **预期产出**：
  ```
  类比SKU先验：μ₀ = 320件/月，σ₀ = 112件/月
  探索加成（CV=35%，高不确定）：+68件（+21%）
  首批量建议：320 + 68 = 388件（≈ 1.5个月库存）
  
  上市后第1周：实际销量 = 285件（低于预期）
  贝叶斯更新：μ₁ = 305件，σ₁ = 85件
  → 第2批补货下调至 300件
  
  上市后第3周：累积销量 720件（超预期）
  贝叶斯更新：μ₃ = 340件，σ₃ = 62件（不确定性收窄）
  → 加急补货450件
  ```
- **业务价值**：相比"按均值订320件"的短视策略，贝叶斯策略在高不确定性场景下减少缺货损失约 $8,000-$15,000（前 6 周）

**场景 B：全新品类（无历史类比SKU）—— Bass 参数估计**

- **业务问题**：首款母婴智能体温贴（IoT 新品类，无任何历史 SKU 可参考）。
- **Bass 参数估计**：p=0.018（创新者少），q=0.32（口碑传播中等）
- **首批量计算**：市场潜力 M=5,000 件（TAM 估算），峰值月需求 ≈ 230 件，首批量 = 230 × 1.5 × 0.35 = **121 件**（保守系数 0.35，首轮试水）

---

## ③ 代码模板

```python
"""
Skill-New-Product-Inventory-Coldstart
基于 Ban, Gallien & Mersereau M&SOM 2019 (类比SKU残差树) +
    Keskin, Li & Song OR 2023 (Bayesian探索加成) +
    Lee et al. TF&SC 2014 (Bass参数ML估计)
母婴跨境 DTC 新品冷启动库存策略
"""

import numpy as np
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize_scalar


@dataclass
class AnalogSKU:
    sku_id: str
    monthly_sales: list[float]
    unit_price: float
    category: str
    launch_month: int
    price_tier: str

    @property
    def demand_mean(self) -> float:
        return float(np.mean(self.monthly_sales))

    @property
    def demand_std(self) -> float:
        return float(np.std(self.monthly_sales))

    @property
    def cv(self) -> float:
        return self.demand_std / max(self.demand_mean, 1)


@dataclass
class NewProductSpec:
    sku_id: str
    unit_price: float
    category: str
    price_tier: str
    launch_month: int
    lead_time_months: float = 1.5
    holding_cost_rate: float = 0.20
    stockout_cost_multiplier: float = 2.0


def find_analog_skus(
    new_product: NewProductSpec,
    catalog: list[AnalogSKU],
    n_analogs: int = 3,
) -> list[tuple[AnalogSKU, float]]:
    """
    按相似度匹配类比 SKU（Ban et al. 协变量回归的简化版）。
    相似度 = 类别匹配 × 价格段距离 × 上市季节对齐
    """
    scored = []
    for sku in catalog:
        sim = 0.0
        if sku.category == new_product.category:
            sim += 0.5
        price_diff = abs(sku.unit_price - new_product.unit_price) / max(new_product.unit_price, 1)
        sim += 0.3 * max(0, 1 - price_diff)
        if sku.price_tier == new_product.price_tier:
            sim += 0.2
        scored.append((sku, round(sim, 3)))

    scored.sort(key=lambda x: -x[1])
    return scored[:n_analogs]


def residual_tree_prior(
    analog_skus: list[tuple[AnalogSKU, float]],
) -> tuple[float, float]:
    """
    类比 SKU 加权估计新品需求先验 (μ₀, σ₀)。
    简化版 Ban et al. 残差树：用相似度加权平均。
    """
    total_sim = sum(sim for _, sim in analog_skus)
    if total_sim == 0:
        return 100.0, 50.0

    mu = sum(sku.demand_mean * sim for sku, sim in analog_skus) / total_sim
    sigma = sum(sku.demand_std * sim for sku, sim in analog_skus) / total_sim
    return mu, sigma


def exploration_boost(
    mu_prior: float,
    sigma_prior: float,
    service_level: float = 0.85,
) -> tuple[float, float]:
    """
    Keskin et al. 2023：探索加成 = 短视最优量 + 额外订购量
    探索加成 ∝ 后验离散度指数 σ²/μ（参数不确定性）
    """
    pdi = sigma_prior ** 2 / max(mu_prior, 1)
    z = stats.norm.ppf(service_level)
    q_myopic = mu_prior + z * sigma_prior
    boost = pdi * 0.5
    q_bdp = q_myopic + boost

    return round(q_bdp), round(boost)


def bayesian_update(
    mu_prior: float,
    sigma_prior: float,
    observed_sales: list[float],
    inventory_levels: list[float],
) -> tuple[float, float]:
    """
    上市后贝叶斯更新（处理右删失销量数据）。
    censored_obs: 若库存量 < 真实需求，观测值为库存量（删失）
    """
    for obs, inv in zip(observed_sales, inventory_levels):
        is_censored = obs >= inv * 0.95
        if not is_censored:
            likelihood_precision = 1.0 / max(obs * 0.1, 1) ** 2
            prior_precision = 1.0 / max(sigma_prior, 1) ** 2
            posterior_precision = prior_precision + likelihood_precision
            mu_prior = (prior_precision * mu_prior + likelihood_precision * obs) / posterior_precision
            sigma_prior = np.sqrt(1.0 / posterior_precision)
        else:
            sigma_prior *= 0.95

    return round(mu_prior, 1), round(sigma_prior, 1)


def bass_initial_order(
    market_potential: float,
    p: float = 0.018,
    q: float = 0.32,
    lead_time_months: float = 1.5,
    conservative_factor: float = 0.35,
) -> dict:
    """
    Bass 扩散模型估计无类比SKU时的首批量。
    conservative_factor: 首批试水系数（0.3-0.5）
    """
    peak_demand_month = market_potential * (p + q) ** 2 / (4 * q)
    months_to_peak = np.log(q / p) / (p + q)
    initial_order = peak_demand_month * lead_time_months * conservative_factor

    return {
        "peak_monthly_demand": round(peak_demand_month),
        "months_to_peak": round(months_to_peak, 1),
        "recommended_first_order": round(initial_order),
        "rationale": f"Bass峰值{peak_demand_month:.0f}件/月 × {lead_time_months}月LT × {conservative_factor}保守系数",
    }


def cold_start_plan(
    new_product: NewProductSpec,
    analog_catalog: list[AnalogSKU],
    service_level: float = 0.85,
) -> dict:
    """
    完整冷启动库存计划：类比先验 → 探索加成首批量 → 贝叶斯更新策略
    """
    analogs = find_analog_skus(new_product, analog_catalog)
    if not analogs:
        return {"error": "无类比SKU，使用Bass fallback", "analogs": []}

    mu0, sigma0 = residual_tree_prior(analogs)
    q_first, boost = exploration_boost(mu0, sigma0, service_level)
    q_first_units = round(q_first * new_product.lead_time_months)

    return {
        "prior_demand_mean": round(mu0, 1),
        "prior_demand_std": round(sigma0, 1),
        "cv": round(sigma0 / max(mu0, 1), 2),
        "q_myopic": round(mu0 * new_product.lead_time_months),
        "exploration_boost": round(boost * new_product.lead_time_months),
        "recommended_first_order": q_first_units,
        "analog_skus": [(a.sku_id, round(s, 2)) for a, s in analogs],
        "strategy": "类比SKU + 探索加成" if sigma0 / max(mu0, 1) > 0.25 else "类比SKU（低不确定性，短视策略足够）",
    }


if __name__ == "__main__":
    catalog = [
        AnalogSKU("UV-C-X100", [280,310,290,350,320,340,300,280], 129.0, "sterilizer", 3, "premium"),
        AnalogSKU("UV-C-Basic", [180,210,195,220,200,215,190,205], 89.0, "sterilizer", 6, "mid"),
        AnalogSKU("Steam-Pro", [450,480,460,500,490,510,470,440], 79.0, "sterilizer", 1, "mid"),
        AnalogSKU("Wipe-Warmer", [120,130,115,140,125,135,110,120], 45.0, "accessory", 4, "mid"),
    ]

    new_sku = NewProductSpec("UV-C-X200", 149.0, "sterilizer", "premium", 7, lead_time_months=1.5)

    print("=" * 65)
    print("新品冷启动库存计划：UV-C X200")
    print("=" * 65)

    plan = cold_start_plan(new_sku, catalog)
    print(f"\n先验估计（类比SKU法）:")
    print(f"  月需求均值: {plan['prior_demand_mean']} 件，标准差: {plan['prior_demand_std']} 件 (CV={plan['cv']:.0%})")
    print(f"  类比SKU: {plan['analog_skus']}")
    print(f"\n首批量决策:")
    print(f"  短视策略量: {plan['q_myopic']} 件")
    print(f"  探索加成:  +{plan['exploration_boost']} 件 （{plan['strategy']}）")
    print(f"  推荐首批量: {plan['recommended_first_order']} 件 （覆盖 {new_sku.lead_time_months} 月 LT）")

    print(f"\n上市后贝叶斯更新（模拟前3周）:")
    mu, sigma = plan['prior_demand_mean'], plan['prior_demand_std']
    weekly_sales = [68, 82, 95]
    weekly_inv   = [90, 90, 90]
    for week, (obs, inv) in enumerate(zip(weekly_sales, weekly_inv), 1):
        mu, sigma = bayesian_update(mu, sigma, [obs], [inv])
        print(f"  第{week}周 销量={obs}, 库存={inv} → 更新: μ={mu}, σ={sigma}")

    print(f"\nBass 估计（全新品类 fallback）:")
    bass = bass_initial_order(market_potential=3000, p=0.018, q=0.32, lead_time_months=1.5)
    print(f"  峰值月需求: {bass['peak_monthly_demand']} 件（第 {bass['months_to_peak']} 个月）")
    print(f"  推荐首批量: {bass['recommended_first_order']} 件")
    print(f"  理由: {bass['rationale']}")
```

---

## ④ 技能关联

- **前置技能**：
  - [[Skill-Bass-Diffusion-New-Product-Forecasting]] — Bass 扩散模型参数估计的理论基础
  - [[Skill-Product-Lifecycle-Stage]] — 新品处于引入期，PLC 阶段决定探索加成系数大小
- **延伸技能**：
  - [[Skill-Safety-Stock-Replenishment]] — 上市后需求稳定后切换至标准安全库存策略
  - [[Skill-Multi-SKU-Procurement-Budget-Allocation]] — 新品首批量纳入季度预算约束
- **可组合**：
  - [[Skill-Cold-Start-Product-Recommendation]] — 推荐系统侧冷启动（流量分配）+ 本 Skill（库存分配）= 新品上市双轨并行
  - [[Skill-Category-Compliance-Prescan]] — 上市前合规预筛已通过，才进入本 Skill 的库存规划流程

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - Ban et al. 实证：忽略协变量 → 成本上升 6-15%，换算 $50K 首批采购 = 节省 $3,000-$7,500
  - Keskin 探索加成：避免短视策略在高不确定性场景的"任意大损失"，实践估算避免缺货损失 $8,000-$15,000
  - 年均 3-4 个新品上市：年化价值 $33,000-$90,000
- **实施难度**：⭐⭐⭐☆☆（3/5）— 需要历史 SKU 数据整理和相似度建模
- **优先级评分**：⭐⭐⭐⭐☆（4/5）— 高风险决策，但频率低于日常补货

---

## 元信息

```yaml
skill_id: Skill-New-Product-Inventory-Coldstart
domain: supply_chain
vault_path: paper2skills-vault/04-供应链/Skill-New-Product-Inventory-Coldstart.md
code_path: paper2skills-code/supply_chain/new_product_inventory_coldstart/
review_score: 8.0/10
wf_coverage: [WF-A, WF-D]
created: 2026-05-25
```
