# Skill Card: CUPED Variance Reduction（方差缩减）

> **领域**: 02-A_B实验 | **类型**: 综合萃取

---

## ① 算法原理

CUPED (Controlled-experiment Using Pre-Experiment Data) 用实验前数据作为协变量，减少实验组间方差，使同样的样本量能检测到更小的效应量。核心公式：$Y_{cuped} = \bar{Y} - \theta(\bar{X} - \mu_X)$，其中 $X$ 是实验前同一用户的指标值，$\theta = \text{Cov}(Y,X)/\text{Var}(X)$。方差缩减率 $\approx 1 - \rho^2_{Y,X}$。若实验前购买金额与实验期购买金额相关系数 $\rho=0.7$，方差缩减 49%。

---

## ② 母婴出海应用案例

A/B 测试吸奶器新详情页——实验前 14 天用户浏览时长作为协变量 $\rho=0.65$→方差缩减 42%→所需样本量从 10,000 降至 5,800。测试周期从 2 周缩短到 10 天。

年化：加速实验迭代，隐性价值 **20-40 万元**。

---

## ③ 代码模板

```python
import numpy as np

def cuped_adjust(Y_treatment, Y_control, X_treatment, X_control):
    theta = np.cov(np.concatenate([Y_treatment, Y_control]), 
                   np.concatenate([X_treatment, X_control]))[0,1] / np.var(np.concatenate([X_treatment, X_control]))
    Yt_adj = Y_treatment - theta * (X_treatment - np.mean(X_treatment))
    Yc_adj = Y_control - theta * (X_control - np.mean(X_control))
    var_reduction = 1 - np.var(Yt_adj - Yc_adj) / np.var(Y_treatment - Y_control)
    return {'adj_diff': np.mean(Yt_adj)-np.mean(Yc_adj), 'var_reduction': var_reduction}

np.random.seed(42)
X = np.random.normal(100,20,2000)
Yt = X*0.7 + np.random.normal(105,8,1000)
Yc = X*0.7 + np.random.normal(100,8,1000)
r = cuped_adjust(Yt, Yc, X[:1000], X[1000:])
print(f"Variance reduction: {r['var_reduction']:.0%}")
assert r['var_reduction'] > 0.3
print("[✓] CUPED 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-AB-Experimental-Design]]
- **组合**：[[Skill-Sequential-AB-Testing]]

---

- **可组合**：[[Skill-Uplift-Modeling]] / [[Skill-Multi-Armed-Bandit]]

## ⑤ 商业价值：20-40 万元 | **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐⭐☆
