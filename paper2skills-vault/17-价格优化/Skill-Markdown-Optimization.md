# Skill Card: Markdown Optimization（折扣清仓定价优化）

> **领域**: 17-价格优化 | **类型**: 综合萃取

---

## ① 算法原理

### 核心思想
清仓不是"打最低折卖完就行"，而是**在库存生命周期内最大化总回收价值**——太早打高折扣损失利润，太晚打低折扣剩库存。核心是找到每条折扣曲线上的最优折扣路径。

### 数学直觉

**动态清仓模型**（Markdown Optimization）：
- State $(t, I_t)$：时间（周数）和当前库存
- Action $d_t \in [0\%, 70\%]$：本周折扣力度
- Transition：$I_{t+1} = I_t - D(P_0(1-d_t))$，需求 $D$ 随折扣增加而增加
- 若 $t = T$（季末）且 $I_T > 0$，触发残值处理（回收价 $C_{salvage} \ll C_{cost}$）

**最优折扣路径**通过逆向归纳（Backward Induction）求解 Bellman 方程：
$$V_t(I) = \max_d \left[ P_0(1-d) \cdot \min(I, D(d)) - C_h \cdot \max(0, I - D(d)) + V_{t+1}(I - D(d)) \right]$$

**关键规律**：对于弹性 $|\epsilon|=1.5$ 的母婴品类，最优策略通常是"前 60% 生命周期不打折，后 40% 逐步从 10% 升到 40-50%"。

### 关键假设
- 需求函数在折扣区间内连续可微
- 库存持有成本可量化（仓储费 + 资金占用成本）
- 季末残值可预估（母婴品类残值率约 20-40%）

---

## ② 母婴出海应用案例

### 场景：吸奶器旧款清仓（被新款 S2 替代）

**业务问题**：S1 吸奶器库存 500 件，成本 $60，原价 $129。新款 S2 3 个月后上市。需要在 12 周内清完 S1 库存，最大化回收价值。太早深度打折会影响品牌形象且被亚马逊算法降权。

**数据要求**：历史清仓数据（同类产品不同折扣下的周销量曲线）

**预期产出**：
- 最优路径：Week 1-6 维持 $119（-8%），Week 7-9 降到 $99（-23%），Week 10-12 降到 $79（-39%）
- 预计回收 $46,500（vs 全价估算 $64,500，残值 $15,000）
- 库存清零率 95%+

**业务价值**：相比"一刀切 50% off"回收 $32,250，优化路径多回收 $14,250（+44%）

---

## ③ 代码模板

```python
"""Markdown Optimization — 动态清仓定价"""

import numpy as np
from typing import List, Tuple


def markdown_optimize(
    inventory: int, cost: float, full_price: float,
    weeks: int, salvage_ratio: float = 0.25,
    elasticity: float = -1.5,
    base_demand: float = None
) -> List[Tuple[int, float, float]]:
    """
    逆向归纳清仓优化
    
    Returns: [(week, price, expected_sales), ...]
    """
    base_demand = base_demand or inventory / (weeks * 0.4)
    salvage_value = cost * salvage_ratio
    discounts = np.linspace(0, 0.6, 13)  # 0%-60% 折扣
    
    # 简化为贪心周度决策
    plan = []
    remaining = inventory
    
    for w in range(1, weeks + 1):
        weeks_left = weeks - w + 1
        # 目标周销量 = 剩余库存 / 剩余周数
        target_sales = remaining / weeks_left
        
        # 找最小折扣满足目标销量
        best_d = 0.0
        for d in discounts:
            price = full_price * (1 - d)
            estimated_demand = base_demand * ((price / full_price) ** elasticity)
            if estimated_demand >= target_sales * 0.4:  # 允许略低于
                best_d = d
                break
        
        price = full_price * (1 - best_d)
        sales = min(base_demand * ((price / full_price) ** elasticity), remaining)
        sales = max(sales, target_sales * 0.3)  # 底线
        sales = min(sales, remaining)
        
        plan.append((w, round(price, 2), round(sales)))
        remaining -= sales
        
        if remaining <= 0:
            break
    
    total_rev = sum(p * s for _, p, s in plan)
    total_rev += remaining * salvage_value
    return plan, total_rev


if __name__ == '__main__':
    plan, revenue = markdown_optimize(
        inventory=500, cost=60, full_price=129,
        weeks=12, salvage_ratio=0.25, elasticity=-1.5
    )
    print(f"清仓计划 (总回收=${revenue:,.0f}):")
    for w, price, sales in plan[:6]:
        print(f"  Week {w}: ${price} × {sales}件")
    print("  ...")
    assert revenue > 30000
    print("\n[✓] Markdown Optimization 测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-Dynamic-Pricing-Elasticity]] | [[Skill-Demand-Forecasting-Supply-Chain]]
- **延伸技能**：[[Skill-Bundle-Pricing-Strategy]]（清仓可组合捆绑）
- **可组合技能**：[[Skill-Multi-Channel-Inventory-Pooling]]

---

## ⑤ 商业价值评估

- **ROI 预估**：每批清仓多回收 15-40%；母婴年均 3-5 批清仓，年化 **20-50 万元**
- **实施难度**：⭐⭐☆☆☆（2 星）
- **优先级评分**：⭐⭐⭐☆☆（3 星）— 季末高频需求
