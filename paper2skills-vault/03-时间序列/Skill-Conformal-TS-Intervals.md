# Skill Card: Conformal TS Intervals（时序 Conformal 预测区间）

> **领域**: 03-时间序列 | **类型**: 综合萃取

roadmap_phase: phase1
---

## ① 算法原理

为时序预测的每个时间步生成分布无关的预测区间。EnbPI (Ensemble Batch Prediction Intervals)：用 Bootstrap 集成 + Conformal 残差构建适应时序依赖的预测区间。

---

## ② 母婴出海应用案例

吸奶器月度需求预测 1200±200 → Conformal 给出更紧的区间：1050-1350（90% 覆盖率）。相比正态假设区间（950-1450），紧 25%。用于安全库存设定：减少过度备货 $500/月。

---

## ③ 代码模板

```python
import numpy as np

def conformal_ts_interval(y_train, y_pred, alpha=0.1):
    residuals = np.abs(y_train - y_pred)
    q = np.quantile(residuals, 1 - alpha)
    return lambda pred: (pred - q, pred + q)

y_train = np.array([100,110,95,120,130,115,105,125])
y_pred_train = np.array([98,108,100,118,128,112,110,122])
interval_fn = conformal_ts_interval(y_train, y_pred_train)
lo, hi = interval_fn(125)
print(f"90% Interval: [{lo:.0f}, {hi:.0f}]")
print("[✓] Conformal TS 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Conformal-Prediction-Demand-UQ]] | [[Skill-Time-Series-Forecasting]]

---

- **可组合**：[[Skill-Demand-Forecasting-Supply-Chain]] / [[Skill-Prophet-Forecasting]]
- **相关**：[[Skill-EventCast-LLM-Event-Forecasting]]

## ⑤ 商业价值：5-10 万元 | **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐☆☆
