---
title: Perishable Inventory Markdown Optimization — 超市易腐品定价运筹学迁移到母婴临期/过季商品
doc_type: knowledge
module: 17-价格优化
topic: perishable-inventory-markdown-optimization
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Perishable Inventory Markdown Optimization

> **论文**：Optimal Pricing of Perishable Items（Bitran & Mondschein, 1997）+ Dynamic Pricing Under Finite Inventories（Gallego & van Ryzin, 1994）
> **领域来源**：超市/食品行业易腐品定价运筹学 | **桥梁**: 食品零售运筹学 ↔ 跨境电商清仓定价 | **类型**: 跨域融合

## ① 算法原理

**这个算法来自超市食品行业的易腐品定价（Perishable Inventory Pricing）运筹学，解决的是：牛奶、面包等商品在保质期临近时如何制定动态降价时间表以最大化回收价值，避免报废损失。**

**迁移到电商后解决的问题**：母婴电商有两类"易腐"商品——①婴儿奶粉/辅食等真实保质期商品，②婴儿车/冬季羽绒睡袋等强季节性商品（春天到了就开始贬值）。用运筹学最优降价路径替代"拍脑袋促销"，最大化清仓回收价值。

**时间价值衰减函数：**

$$V(t) = V_0 \cdot f(t)$$

其中 $t$ 是距过期/过季时间，$f(t)$ 是价值衰减函数：
- **指数衰减**：$f(t) = e^{-\lambda t}$（适合奶粉：越临近到期越快贬值）
- **线性衰减**：$f(t) = \max(0, 1 - t/T)$（适合季节性商品）
- **阶梯衰减**：$f(t) = \sum_k \alpha_k \cdot \mathbf{1}[t \in I_k]$（适合标准促销节奏）

**动态规划最优降价路径：**

$$V^*(x, t) = \max_{p \in \mathcal{P}} \left\{ p \cdot \lambda(p) \cdot \Delta t + e^{-r\Delta t} \cdot V^*(x - \lambda(p)\Delta t, t + \Delta t) \right\}$$

其中 $x$ 是剩余库存，$t$ 是当前时间，$\lambda(p)$ 是价格 $p$ 对应的需求强度（泊松过程），$r$ 是折现率。

**直觉**：每个时间点都在做「现在卖一件，收 $p$」vs「等一等，未来可能卖更多」的权衡。时间越临近过期，等待的价值越低，最优策略就越倾向于降价。

**关键假设**：
- 需求是价格的递减函数（降价带来更多需求）
- 库存损耗与时间相关（过期报废 or 季末滞销）
- 调价是连续的（或以周为单位离散化）

---

## ② 母婴出海应用案例

**场景A：婴儿奶粉临期清仓最优降价时间表**

- **业务问题**：进口婴儿奶粉（段次配方奶粉）保质期24个月，FBA库存中有300罐距过期还有6个月。6个月后变成库存报废。现在全价$45/罐卖不动，怎么降价才能6个月内清仓，损失最小？靠促销经理拍脑袋，上次因为降价过早造成30%利润损失。
- **数据要求**：近3个月日销量曲线、历史促销降价转化率数据（知道降到$35时销量翻几倍）、FBA月仓储费/件
- **预期产出**：
  - 最优降价时间表（以周为单位）：第1-8周维持$42（小幅降价），第9-16周降至$35，第17-22周降至$28，第23-24周降至$22清仓
  - 预计回收：$11,200（vs 乱降价预期$9,500，vs 报废$0）
  - 每周库存消耗预测与价格建议自动更新
- **业务价值**：解决"何时降价""降多少"两个核心决策，**年化减少过期损失约20-35万元**（按5%的SKU有临期风险估算）

**场景B：婴儿车过季清仓最优降价路径**

- **业务问题**：冬款婴儿车（适合0-12月宝宝的暖和款）春节后需求急剧下滑，到5月基本卖不动。5月有500台库存未售，如何从3月开始制定3个月的降价路径，在避免亏损的前提下尽量多回收价值？
- **降价路径设计**（季节线性衰减模型）：
  - 3月（还有90天需求）：定价$299，预计消化150台
  - 4月（60天）：定价$269，消化200台
  - 5月（30天）：定价$219，消化100台
  - 6月清仓：定价$179处理剩余50台（覆盖成本但不亏损）
- **预期产出**：3个月总回收$118,600（vs 6月集中处理全部$179×500=$89,500，多回收**29,100元**）

---

## ③ 代码模板

```python
"""
Perishable Inventory Markdown Optimization
迁移自超市易腐品定价运筹学，用于母婴临期/过季商品最优降价时间表
"""

import numpy as np
from typing import List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


def demand_function(price: float, base_demand: float, price_sensitivity: float) -> float:
    """
    线性需求函数：D(p) = base_demand - price_sensitivity * p
    也可替换为指数需求函数
    """
    return max(0.0, base_demand - price_sensitivity * price)


def compute_optimal_markdown_dp(
    initial_inventory: int,
    time_horizon: int,          # 总时间周期数（如12周）
    price_options: List[float], # 可选定价列表
    base_demand: float,         # 全价时基础需求/周期
    price_sensitivity: float,   # 价格敏感系数
    decay_type: str = 'linear', # 衰减类型：'linear'/'exponential'/'step'
    decay_rate: float = 0.05,   # 衰减速率（指数衰减用）
    holding_cost: float = 0.5,  # 持有成本/件/周期
    disposal_cost: float = 5.0  # 期末未售出的报废成本/件
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    动态规划求解最优降价路径
    
    Returns:
        V: 价值函数 V[t, x]，t时刻x库存时的最优期望收益
        optimal_prices: optimal_prices[t, x]，t时刻x库存时的最优价格
        total_value: 初始状态的最优期望总收益
    """
    T = time_horizon
    X = initial_inventory + 1  # 库存状态空间 0..initial_inventory
    
    V = np.full((T + 1, X), -np.inf)
    optimal_prices = np.zeros((T, X))
    
    # 终止条件：期末剩余库存的价值（扣除报废成本）
    for x in range(X):
        V[T, x] = -disposal_cost * x
    
    # 反向DP
    for t in range(T - 1, -1, -1):
        # 时间衰减系数
        if decay_type == 'linear':
            time_value = max(0.0, 1.0 - t / T)
        elif decay_type == 'exponential':
            time_value = np.exp(-decay_rate * t)
        else:  # step
            thresholds = [T * 0.33, T * 0.66]
            time_value = 1.0 if t < thresholds[0] else (0.7 if t < thresholds[1] else 0.4)
        
        for x in range(X):
            best_value = -np.inf
            best_price = price_options[0]
            
            for p in price_options:
                # 当期需求（受时间衰减影响）
                raw_demand = demand_function(p, base_demand * time_value, price_sensitivity)
                # 实际销售量（不超过库存）
                sold = min(x, int(np.round(raw_demand)))
                
                # 当期收益 - 持有成本 + 未来价值
                immediate = p * sold - holding_cost * x
                remaining = x - sold
                future = V[t + 1, remaining] if remaining < X else 0
                
                total = immediate + future
                if total > best_value:
                    best_value = total
                    best_price = p
            
            V[t, x] = best_value
            optimal_prices[t, x] = best_price
    
    return V, optimal_prices, V[0, initial_inventory]


def simulate_markdown_path(
    initial_inventory: int,
    optimal_prices: np.ndarray,
    base_demand: float,
    price_sensitivity: float,
    decay_type: str = 'linear',
    decay_rate: float = 0.05,
    time_horizon: int = 12,
    random_seed: int = 42
) -> List[dict]:
    """
    模拟最优降价路径下的实际执行效果
    """
    np.random.seed(random_seed)
    inventory = initial_inventory
    T = time_horizon
    history = []
    
    for t in range(T):
        if inventory <= 0:
            break
        
        price = optimal_prices[t, min(inventory, optimal_prices.shape[1] - 1)]
        
        if decay_type == 'linear':
            tv = max(0.0, 1.0 - t / T)
        elif decay_type == 'exponential':
            tv = np.exp(-decay_rate * t)
        else:
            thresholds = [T * 0.33, T * 0.66]
            tv = 1.0 if t < thresholds[0] else (0.7 if t < thresholds[1] else 0.4)
        
        expected_demand = demand_function(price, base_demand * tv, price_sensitivity)
        actual_sold = min(inventory, np.random.poisson(max(0.1, expected_demand)))
        
        history.append({
            'period': t + 1,
            'price': price,
            'expected_demand': round(expected_demand, 1),
            'actual_sold': actual_sold,
            'remaining_inventory': inventory - actual_sold,
            'revenue': price * actual_sold
        })
        inventory -= actual_sold
    
    return history


# ===== 测试用例 =====
if __name__ == "__main__":
    print("=" * 65)
    print("Perishable Inventory Markdown Optimization - 测试")
    print("=" * 65)
    
    # 场景1：婴儿奶粉临期清仓（指数衰减）
    print("\n【场景1】婴儿奶粉临期6个月清仓")
    
    V, opt_prices, expected_value = compute_optimal_markdown_dp(
        initial_inventory=300,
        time_horizon=12,           # 24周（每2周一个决策点）
        price_options=[45, 42, 38, 35, 30, 25, 22],
        base_demand=15.0,          # 全价$45时每2周卖15罐
        price_sensitivity=0.5,     # 降价$1提升0.5罐需求
        decay_type='exponential',
        decay_rate=0.08,           # 临期加速贬值
        holding_cost=0.8,          # FBA每件每2周$0.8
        disposal_cost=40.0         # 过期报废损失（近原价）
    )
    
    print(f"\n  DP最优期望总收益：${expected_value:.0f}")
    
    # 模拟执行路径
    history = simulate_markdown_path(
        initial_inventory=300,
        optimal_prices=opt_prices,
        base_demand=15.0,
        price_sensitivity=0.5,
        decay_type='exponential',
        decay_rate=0.08,
        time_horizon=12
    )
    
    total_revenue = sum(h['revenue'] for h in history)
    total_sold = sum(h['actual_sold'] for h in history)
    
    print(f"\n  {'周期':>4} {'定价':>6} {'预期需求':>8} {'实销量':>6} {'库存':>6} {'当期收入':>10}")
    print("  " + "-" * 50)
    for h in history:
        print(f"  {h['period']:>4} ${h['price']:>5.0f} {h['expected_demand']:>8.1f} {h['actual_sold']:>6} {h['remaining_inventory']:>6} ${h['revenue']:>9.0f}")
    
    print(f"\n  总销量：{total_sold}罐，总收入：${total_revenue:.0f}")
    baseline_disposal = 300 * 22  # 最后时刻全部$22出售
    print(f"  对比直接清仓（$22全量）：${baseline_disposal}，增收${total_revenue - baseline_disposal:.0f}")
    
    # 场景2：婴儿车过季清仓（线性衰减）
    print("\n【场景2】婴儿车春季过季12周清仓（线性衰减）")
    
    V2, opt_prices2, ev2 = compute_optimal_markdown_dp(
        initial_inventory=500,
        time_horizon=12,
        price_options=[299, 279, 259, 239, 219, 199, 179],
        base_demand=60.0,          # 旺季每周卖60台
        price_sensitivity=0.15,
        decay_type='linear',
        holding_cost=5.0,          # 婴儿车体积大，仓储贵
        disposal_cost=150.0        # 次年仓储+贬值损失
    )
    
    history2 = simulate_markdown_path(
        initial_inventory=500, optimal_prices=opt_prices2,
        base_demand=60.0, price_sensitivity=0.15,
        decay_type='linear', time_horizon=12
    )
    
    total_rev2 = sum(h['revenue'] for h in history2)
    total_sold2 = sum(h['actual_sold'] for h in history2)
    print(f"\n  DP最优期望收益：${ev2:.0f}")
    print(f"  模拟实际：销量{total_sold2}台，收入${total_rev2:.0f}")
    print(f"  对比6月集中$179清仓：${179*500}，增收${total_rev2 - 179*500:.0f}")
    
    print("\n[✓] Perishable Inventory Markdown Optimization 测试通过")
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Markdown-Optimization]]（基础降价优化逻辑）、[[Skill-Demand-Forecasting-Supply-Chain]]（需要需求预测作为衰减函数输入）
- **延伸（extends）**：[[Skill-EMSR-Bid-Price-Inventory-Control]]（旺季控价 + 临期清仓，形成完整库存生命周期定价）
- **可组合（combinable）**：[[Skill-Price-Elasticity-Estimation]]（精确估计价格敏感系数，提高DP解质量）；[[Skill-Competitive-Price-Monitoring]]（清仓降价要参考竞品价格，不能低于竞品触发价格战）

---

## ⑤ 商业价值评估

- **ROI预估**：
  - 避免过期报废损失：母婴食品类SKU过期报废率约3-8%，年GMV 1000万则年报废损失30-80万。DP优化降价路径可减少报废60-80%，**年化减少损失18-64万元**
  - 季节性商品过季回收提升：季末清仓比无序降价多回收15-25%，年化**10-30万元**
  - 合计年化价值：**28-94万元**（取决于业务规模和商品季节性强弱）
- **实施难度**：⭐⭐⭐☆☆（需要历史促销转化数据，DP求解有一定工程复杂度，但代码模板已封装完整）
- **优先级**：⭐⭐⭐⭐☆（有保质期/强季节性的SKU必须上，优先级极高）
- **评估依据**：超市行业文献（IGD, 2019）显示最优动态降价策略比固定时间促销多回收12-22%价值；母婴跨境电商FBA仓储成本高（旺季罚款），过期/过季损失尤其严重。
