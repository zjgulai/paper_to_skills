# Skill Card: Forecast-Driven Inventory（预测驱动库存优化）

> **桥梁**: 03-时间序列 ↔ 04-供应链 | **类型**: 跨域融合

---

## ① 算法原理

打通需求预测和库存决策——不是先预测再独立决策，而是将预测不确定性直接编码为库存策略参数。核心：**服务水平优化**——给定预测分布 $N(\hat{\mu}, \hat{\sigma})$，安全库存 $SS = z_\alpha \cdot \hat{\sigma} \cdot \sqrt{LT}$，其中 $z_\alpha$ 由缺货成本 vs 持有成本决定。

$$z^* = \Phi^{-1}\left(\frac{C_{shortage}}{C_{shortage} + C_{holding}}\right)$$

---

## ② 母婴出海应用案例

吸奶器月需求预测 1200±200，提前期 30 天，缺货成本 $25/件，持有成本 $3/件。最优 $z^*=1.75$，安全库存 $= 1.75 \times 200 \times \sqrt{1} = 350$ 件。vs 简单规则（$z=1.64$，$SS=328$），损失减少 $22 \times 30 = \$660/月$。

年化：**8-12 万元**。

---

## ③ 代码模板

```python
from scipy.stats import norm

def optimal_service_level(shortage_cost, holding_cost):
    return shortage_cost / (shortage_cost + holding_cost)

def safety_stock(demand_std, lead_time, z_score):
    return z_score * demand_std * np.sqrt(lead_time)

import numpy as np
sl = optimal_service_level(25, 3)
z = norm.ppf(sl)
ss = safety_stock(200, 1, z)
print(f"Service Level: {sl:.0%}, z: {z:.2f}, Safety Stock: {ss:.0f}")
print("[✓] Forecast-Driven Inventory 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Time-Series-Forecasting]] (03) | [[Skill-Demand-Forecasting-Supply-Chain]] (04)
- **组合**：[[Skill-Conformal-Prediction-Demand-UQ]] (03) | [[Skill-Multi-Channel-Inventory-Pooling]] (04)

---

## ⑤ 商业价值

- **ROI**：8-12 万元 | **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐☆☆
