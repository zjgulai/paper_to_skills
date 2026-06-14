# Skill Card: Multivariate Cointegration（多变量协整 VECM）

> **领域**: 03-时间序列 | **类型**: 综合萃取

roadmap_phase: phase1
---

## ① 算法原理

多个时序共享长期均衡关系（如吸奶器销量←→配件销量）。Johansen 协整检验确定协整向量数量，VECM 建模短期动态调整 + 长期均衡约束。$\Delta Y_t = \Pi Y_{t-1} + \sum \Gamma_i \Delta Y_{t-i} + \epsilon_t$，其中 $\Pi = \alpha\beta'$（$\beta$ 是协整向量，$\alpha$ 是调整速度）。

---

## ② 母婴出海应用案例

吸奶器月销 1000 件，法兰月销 3000 件。Johansen 检验确认协整关系（$p<0.01$），VECM 预测：当吸奶器销量 +10% 时，法兰销量将在 2 周内 +6.5%。用于供应链联动补货。

---

## ③ 代码模板

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
import numpy as np

def test_cointegration(data: np.ndarray, det_order=0, k_ar_diff=1):
    """data: (n_obs, n_vars)"""
    result = coint_johansen(data, det_order, k_ar_diff)
    return {'trace_stat': result.lr1, 'n_coint': sum(result.lr1 > result.cvt[:, 0])}

# test
np.random.seed(42)
pump = np.cumsum(np.random.randn(100)) + 1000
flange = pump * 3 + np.cumsum(np.random.randn(100)*0.5)
r = test_cointegration(np.column_stack([pump, flange]))
print(f"Cointegrating vectors: {r['n_coint']}")
assert r['n_coint'] >= 1
print("[✓] Multivariate Cointegration 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Time-Series-Forecasting]] | [[Skill-Conformal-Prediction-Demand-UQ]]
- **组合**：[[Skill-Category-Trend-Forecasting]]

---
- **相关**：[[Skill-EventCast-LLM-Event-Forecasting]]
- **相关**：[[Skill-Demand-Forecasting-Supply-Chain]]

## ⑤ 商业价值：8-15 万元 | **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐☆☆
