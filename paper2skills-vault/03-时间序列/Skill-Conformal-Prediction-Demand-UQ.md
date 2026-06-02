# Skill Card: Conformal Prediction Demand UQ（需求预测不确定性量化）

> **论文**: Conformal PID Control for Time Series Prediction (arXiv:2307.16895, NeurIPS 2023)  
> **辅论文**: CopulaCPTS (arXiv:2212.03281, ICLR 2024)  
> **领域**: 03-时间序列 | **服务工作流**: WF-A (P15)

---

## ① 算法原理

### 核心思想
传统需求预测只给点估计（"下月卖 1000 件"），业务需要区间估计（"95% 置信区间: 850-1200 件"）。Conformal Prediction 提供分布无关的、有限样本有效的预测区间，无需假设误差分布。

### 数学直觉

**Conformal PID Control**（arXiv:2307.16895）：将 conformal prediction 与 PID 控制论融合：

- **P（比例）**：Quantile tracking — 跟踪滚动误差的 $\alpha$ 分位数，维持目标覆盖率
- **I（积分）**：累积覆盖误差修正 — 如果过去 10 天覆盖率低于目标，上调区间宽度
- **D（微分）**：Scorecasting — 用轻量模型预测下一步的 nonconformity score，提前适应季节性/趋势变化

$$\hat{C}_t(x) = [\hat{f}(x) - q_t \cdot s(x), \hat{f}(x) + q_t \cdot s(x)]$$
其中 $q_t$ 由 PID 控制器动态调整，$s(x)$ 为预测的 nonconformity score。

**CopulaCPTS（多步扩展）**：用 Copula 对多步预测误差的时序依赖建模，构建联合置信区间，避免逐步独立的 Bonferroni correction 过于保守。

### 关键假设
- 数据可交换性（exchangeability）— 时序场景下通过滑动窗口近似保证
- PID 参数需要按品类特性调优（母婴季节性强的品类需要更大的 D 分量）

---

## ② 母婴出海应用案例

### 场景一：吸奶器月度需求预测的安全库存设定

**业务问题**：Prophet 预测下月销量 1200 件。但点估计不能直接用于安全库存——需要 90% 置信区间来决定备货量。

**数据要求**：24 个月月度销量 × 10 SKU。Conformal PID 自适应季节性（Q4 旺季区间自然变宽）

**预期产出**：
- 90% 预测区间：[1020, 1420]（vs Prophet 点估计 1200）
- 安全库存建议：按区间上界 1420 备货，避免缺货损失
- 覆盖率追踪：实际值落在区间内的频率维持在 90%±3%

**业务价值**：缺货损失 $15-25/件 × 缺口 100 件/月 = $1,500-2,500/月 避免；同时不过度备货（区间下界防止库存积压）。年化 **30-50 万元**

### 场景二：新品冷启动的 14 天滚动预测

**业务问题**：新品只有 14 天数据，传统预测方法不可靠。Conformal 的有限样本有效性在此场景下优势明显。

**数据要求**：14 天日销量 + 同类老品的历史数据作为先验

**预期产出**：宽但诚实的预测区间——新品不确定性格外大，但至少量化了"我们有多不确定"

**业务价值**：新品阶段避免因过度乐观备货导致的库存积压（均价 $30 × 500 件过剩 = $15,000）

---

## ③ 代码模板

```python
"""
Conformal Prediction for Demand UQ — PID 自适应预测区间
基于 Conformal PID Control (arXiv:2307.16895)
"""

import numpy as np
from typing import Tuple, List


class ConformalPID:
    """Conformal PID 控制器 — 自适应预测区间"""
    
    def __init__(self, alpha: float = 0.1,  # 目标误覆盖率
                 kp: float = 0.5, ki: float = 0.1, kd: float = 0.05):
        self.alpha = alpha
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.quantile = 1.0  # 初始 quantile
    
    def update(self, actual: float, predicted: float, 
               score: float) -> float:
        """PID 更新 quantile，返回调整后的预测区间半宽"""
        error = (1 - self.alpha) - (abs(actual - predicted) <= self.quantile * score)
        self.integral_error = 0.9 * self.integral_error + error
        derivative = error - self.prev_error
        adjustment = self.kp * error + self.ki * self.integral_error + self.kd * derivative
        self.quantile = max(0.5, min(3.0, self.quantile + adjustment))
        self.prev_error = error
        return self.quantile * score


def conformal_forecast_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    alpha: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    生成 conformal 预测区间
    
    Returns:
        (lower_bounds, upper_bounds, coverages)
    """
    pid = ConformalPID(alpha=alpha)
    n = len(y_true)
    lower = np.zeros(n)
    upper = np.zeros(n)
    coverages = []
    
    for t in range(n):
        half_width = pid.update(y_true[t], y_pred[t], scores[t])
        lower[t] = y_pred[t] - half_width
        upper[t] = y_pred[t] + half_width
        covered = 1.0 if lower[t] <= y_true[t] <= upper[t] else 0.0
        coverages.append(covered)
    
    # 滚动覆盖率
    window = 30
    rolling_cov = [np.mean(coverages[max(0,i-window):i+1]) 
                   for i in range(n)]
    
    return lower, upper, rolling_cov


# ============ 测试 ============

if __name__ == '__main__':
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    
    # 模拟带季节性的需求
    base = 100 + 20 * np.sin(2 * np.pi * t / 12)  # 12 月周期
    noise = np.random.normal(0, 10, n)
    y_true = base + noise
    y_pred = base + np.random.normal(0, 8, n)  # 预测带噪声
    
    # Nonconformity scores（简化为预测误差的滚动 std）
    window = 14
    scores = np.array([np.std(y_true[max(0,i-window):i+1] - y_pred[max(0,i-window):i+1]) 
                       if i >= 3 else 5.0 for i in range(n)])
    
    lower, upper, coverages = conformal_forecast_intervals(y_true, y_pred, scores, alpha=0.1)
    
    actual_cov = np.mean([1 if lower[i] <= y_true[i] <= upper[i] else 0 for i in range(n)])
    print(f"PID Conformal: 目标覆盖率=90%, 实际覆盖率={actual_cov:.1%}")
    
    # 验证覆盖率在合理范围
    assert 0.80 <= actual_cov <= 0.98, f"Coverage {actual_cov:.1%} out of expected range"
    print("\n[✓] Conformal Prediction Demand UQ 测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-Time-Series-Forecasting]] | [[Skill-Prophet-Forecasting]]
- **延伸技能**：[[Skill-Hierarchical-Demand-Forecasting-Reconciliation]]（分层预测 + 分层不确定性的联合建模）
- **可组合技能**：[[Skill-Demand-Forecasting-Supply-Chain]] | [[Skill-SSBC-Small-Sample-Conformal]]

---
- **相关技能**：[[Skill-Multivariate-Cointegration]]
- **相关技能**：[[Skill-Forecast-Driven-Inventory]]
- **相关技能**：[[Skill-Conformal-TS-Intervals]]

## ⑤ 商业价值评估

- **ROI 预估**：减少缺货损失 $1,500-2,500/月 + 减少过度备货积压；年化 **30-50 万元**
- **实施难度**：⭐⭐☆☆☆（2 星）— PID Conformal 有开源实现，即插即用
- **优先级评分**：⭐⭐⭐☆☆（3 星）— 需求预测的第二阶能力（先有点估计，再要区间估计）
- **评估依据**：NeurIPS 2023 顶级团队（Angelopoulos/Candès/Tibshirani），代码已开源 pip 可用
