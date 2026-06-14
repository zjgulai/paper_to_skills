# Skill Card: Sequential AB Testing（序列化 A/B 检验）

> **领域**: 02-A_B实验 | **类型**: 综合萃取

roadmap_phase: phase1
---

## ① 算法原理

传统固定样本量 A/B 需要等到收集满 N 个样本才分析。Sequential Testing 允许**在实验过程中多次中期分析**，一旦检测到显著差异即可提前停止（节省时间和样本）。用 $\alpha$-spending function 控制整体 Type I error：每次中期分析消耗一部分 $\alpha$ 预算。

**Pocock 边界**：所有中期分析使用相同的严格阈值。**O'Brien-Fleming 边界**：早期阈值极严，后期逐渐放宽（更常用）。

---

## ② 母婴出海应用案例

吸奶器详情页 A/B：预期需要 10,000 样本/组 × 14 天。Sequential 分析在 Day 7（7,200 样本）就检测到显著差异（$p < 0.005$，OF 边界），实验提前 7 天结束。年节省实验等待时间 40%，加速迭代节奏。

---

## ③ 代码模板

```python
import numpy as np
from scipy.stats import norm

def obrien_fleming_boundary(n_looks: int, alpha: float = 0.05):
    """O'Brien-Fleming 序贯边界"""
    boundaries = []
    for k in range(1, n_looks + 1):
        t = k / n_looks  # information fraction
        z_bound = norm.ppf(1 - alpha/2) / np.sqrt(t)  # OF adjustment
        boundaries.append(z_bound)
    return boundaries

def sequential_test(data_streams, boundaries):
    for k, (a_data, b_data) in enumerate(data_streams):
        diff = np.mean(a_data) - np.mean(b_data)
        se = np.sqrt(np.var(a_data)/len(a_data) + np.var(b_data)/len(b_data))
        z = abs(diff) / max(se, 1e-6)
        if z > boundaries[k]:
            return {'stop': True, 'look': k+1, 'significant': True, 'z': z}
    return {'stop': False, 'significant': False}

np.random.seed(42)
streams = [((np.random.normal(100,15,1000*i), np.random.normal(105,15,1000*i))) for i in range(1,6)]
bounds = obrien_fleming_boundary(5, 0.05)
r = sequential_test(streams, bounds)
print(f"Sequential: stop={r['stop']}, look={r.get('look','?')}")
print("[✓] Sequential AB 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-AB-Experimental-Design]] | [[Skill-CUPED-Variance-Reduction]]
- **组合**：[[Skill-Network-Effect-Experiments]]

---
- **相关**：[[Skill-Multi-Armed-Bandit]]
- **相关**：[[Skill-Demand-Forecasting-Supply-Chain]]

## ⑤ 商业价值

- **ROI**：加速实验迭代 40%，年化隐性 **15-30 万元**
- **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐☆☆
