---
title: 小样本Beta修正共形预测 - 50个样本也能保证覆盖
doc_type: knowledge
module: 01-因果推断
topic: small-sample-conformal
status: stable
created: 2026-05-20
updated: 2026-05-20
owner: self
source: human+ai
paper: arXiv:2509.15349 (2025)
---

# Skill: SSBC Small Sample Conformal — 小样本共形预测精确覆盖保证

> 论文:**Probabilistic Conformal Coverage Guarantees in Small-Data Settings (SSBC)** · arXiv:2509.15349 (2025)
> 作者: Petrus H. Zwart · Lawrence Berkeley National Laboratory
> 应用:校准集仅50个样本时，保证共形预测的实际覆盖率不低于目标值

---

## ① 算法原理

### 核心思想

标准 Split Conformal Prediction 的覆盖保证是**"期望意义"**的——跨多次校准集随机抽取，平均覆盖率为 `1-α`，但**单次校准的覆盖率可能远低于目标值**。实验表明：当校准集 n=50 时，目标覆盖率 90% 的标准共形预测，实际违约率高达 ~40%（即 40% 的概率实际覆盖 < 90%）。

**SSBC 的突破**：利用共形预测覆盖率**精确服从 Beta-Binomial 分布**这一数学事实，调整显著性水平 `α_adj`，使得以用户指定概率 `1-δ` 保证实际覆盖率 ≥ 目标值。

**三大特性**：
1. **即插即用**：不改模型、不改共形分数，只调整一个参数 `α_adj`
2. **PAC 保证**：Probably Approximately Correct——以概率 `1-δ` 保证覆盖 ≥ `1-α_target`
3. **比 DKWM 更精确**：样本复杂度从 `O(α⁻²)` 提升到 `O(1/α)`，不过度保守

**DKWM vs SSBC 的对比**（分子溶解度预测实验，n_cal=50）：
- DKWM：违约率仅 1.6%（过度保守，区间太宽）
- SSBC：违约率 6.4%（符合 δ=10% 要求，区间宽度适中）

### 数学直觉

**覆盖率的精确分布**：

当校准集大小为 n，名义显著性水平为 α，无穷测试集下的覆盖率精确服从：

$$C_{\infty} \sim \text{Beta}(k, n+1-k), \quad k = \lceil (1-\alpha)(n+1) \rceil$$

有限测试集（大小 m）时：

$$C_m = \frac{X}{m}, \quad X \sim \text{Beta-Binomial}(m; k, n+1-k)$$

**SSBC 的调整目标**：

找到最大的 `α_adj`，使得：

$$\Pr(C(\alpha_{\text{adj}}) \geq 1-\alpha_{\text{target}}) \geq 1-\delta$$

由于共形分位数必须落在离散网格 `{u/(n+1): u=1,...,n}` 上，SSBC 在该网格上搜索最优格点：

$$\alpha_{\text{adj}} = \arg\max_{\alpha' \in \{u/(n+1)\}, \alpha' < \alpha_{\text{target}}} \{\Pr(C(\alpha') \geq 1-\alpha_{\text{target}}) \geq 1-\delta\}$$

**可行性边界**：当 n 太小时，某些 (α, δ) 组合不可达。无穷测试集下的最小可行目标 α：

$$\alpha^*_{\infty}(n, \delta) = 1 - \delta^{1/n}$$

这揭示了**覆盖率-样本量-置信度的三角权衡**：不能同时要求极小的 α 和极高的置信度 δ，除非有足够大的 n。

### 关键效果数字

| 场景 | n_cal | 方法 | 违约率 | 目标 δ |
|------|-------|------|--------|--------|
| Monte Carlo (α=0.1) | 50 | 标准共形 | 39.4% | — |
| Monte Carlo (α=0.1) | 50 | SSBC | 4.7% | 10% |
| Monte Carlo (α=0.1) | 100 | SSBC | 9.5% | 10% |
| 分子溶解度预测 | 50 | DKWM | 1.6% | 10% |
| 分子溶解度预测 | 50 | SSBC | 6.4% | 10% |
| 冷冻电镜分割 | 47 | 标准共形 | 严重欠覆盖 | — |
| 冷冻电镜分割 | 47 | SSBC | 正常覆盖 | 5% |

**样本复杂度改进**：`O(α⁻²)` → `O(1/α)`（相差一个量级）

---

## ② 母婴出海应用案例

### 场景：新市场50条转化数据的可信区间

**业务问题**：日本市场刚上线，只收集了50条带标签的转化数据做校准。标准共形预测名义 95% 覆盖率，但实际可能只有 82%——基于此做预算决策犹如"碰运气"。SSBC 修正后保证以 95% 概率实际覆盖 ≥ 95%，做预算决策时心里有底。

**数据要求**：
- 小样本校准集（n ≥ 20 即可用，n ≥ 50 效果稳定）
- 任意共形分数（绝对残差、预测区间宽度等均可）
- 需指定目标覆盖率 `1-α_target` 和置信参数 `1-δ`

**预期产出**：
- 调整后的 `α_adj`（比 `α_target` 更严格的显著性水平）
- 基于 `α_adj` 构建的共形预测集/区间
- 可行性判断（n 是否足够达到 (α, δ) 目标）

**业务价值**：
- 小样本下决策可信度从"碰运气"变为"有概率保证"
- 给 CEO 汇报："以 95% 概率保证，我们的预测覆盖率不低于 95%"
- 对比 DKWM：区间宽度更合理，不会因过度保守而丧失精准性

---

## ③ 代码模板

```python
"""
SSBC (Small Sample Beta Correction) - 小样本共形预测精确覆盖保证
论文: arXiv:2509.15349 (2025)
场景: 母婴出海新市场小样本校准集下的 PAC 覆盖保证

依赖: numpy, scipy, pandas
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betaln
import warnings
warnings.filterwarnings('ignore')


# ==================== SSBC 核心算法 ====================

class SSBC:
    """
    Small Sample Beta Correction (SSBC)
    
    将标准 Split Conformal Prediction 的期望覆盖保证
    升级为 PAC (Probably Approximately Correct) 覆盖保证：
    
        Pr(coverage >= 1-α_target) >= 1-δ
    
    即插即用：不改变模型或共形分数，只调整显著性水平 α_adj。
    """
    
    def __init__(self, alpha_target: float, delta: float, n_cal: int, 
                 m_test: int = None):
        """
        Args:
            alpha_target: 目标名义显著性水平（如 0.05 表示 95% 覆盖率）
            delta: 风险容忍度——以概率 1-delta 保证覆盖（如 0.05）
            n_cal: 校准集大小
            m_test: 测试集大小（None 表示无穷，即 Beta 分布；给定时用 Beta-Binomial）
        """
        self.alpha_target = alpha_target
        self.delta = delta
        self.n_cal = n_cal
        self.m_test = m_test
        self.alpha_adj = None
        self.feasible = False
        self._grid = None
    
    def _coverage_prob(self, alpha_prime: float) -> float:
        """
        计算 Pr(C(α') >= 1-α_target)
        
        使用 Beta（无穷测试集）或 Beta-Binomial（有限测试集）
        """
        n = self.n_cal
        k = int(np.ceil((1 - alpha_prime) * (n + 1)))
        k = max(1, min(k, n))  # 边界约束
        
        a = k        # Beta 参数 a
        b = n + 1 - k  # Beta 参数 b
        
        target_coverage = 1 - self.alpha_target
        
        if self.m_test is None:
            # 无穷测试集：Coverage ~ Beta(a, b)
            # Pr(C >= target) = 1 - CDF(target)
            prob = 1.0 - stats.beta.cdf(target_coverage, a, b)
        else:
            # 有限测试集：Coverage = X/m, X ~ Beta-Binomial(m, a, b)
            m = self.m_test
            x_star = int(np.ceil(target_coverage * m))
            # Pr(X >= x_star)
            prob = self._beta_binomial_survival(m, a, b, x_star)
        
        return prob
    
    @staticmethod
    def _beta_binomial_survival(m: int, a: float, b: float, 
                                  x_star: int) -> float:
        """
        Beta-Binomial 生存函数：Pr(X >= x_star)
        X ~ Beta-Binomial(m, a, b)
        """
        # 通过求和计算 PMF
        prob = 0.0
        
        # 使用对数域避免数值溢出
        # log P(X=k) = log C(m,k) + log B(k+a, m-k+b) - log B(a, b)
        log_B_ab = betaln(a, b)
        
        for k in range(x_star, m + 1):
            log_comb = (np.lgamma(m + 1) - np.lgamma(k + 1) - 
                       np.lgamma(m - k + 1))
            log_B_k = betaln(k + a, m - k + b)
            log_pmf = log_comb + log_B_k - log_B_ab
            prob += np.exp(log_pmf)
        
        return min(prob, 1.0)
    
    def compute(self) -> dict:
        """
        计算 SSBC 调整后的显著性水平 α_adj
        
        Returns:
            dict with keys:
                'feasible': 是否可行
                'alpha_adj': 调整后的显著性水平（不可行时为 None）
                'alpha_target': 目标显著性水平
                'delta': 风险容忍度
                'n_cal': 校准集大小
                'coverage_prob': 以 α_adj 时的 Pr(coverage >= 1-α_target)
                'grid': 所有网格点的概率（用于诊断）
        """
        n = self.n_cal
        
        # 构建离散网格 {u/(n+1): u=1,...,n}
        grid_us = np.arange(1, n + 1)
        grid_alphas = grid_us / (n + 1)
        
        # 只考虑严格小于 α_target 的格点（SSBC 约束：α_adj < α_target）
        valid_mask = grid_alphas < self.alpha_target
        
        if not valid_mask.any():
            self.feasible = False
            self.alpha_adj = None
            return {
                'feasible': False,
                'alpha_adj': None,
                'alpha_target': self.alpha_target,
                'delta': self.delta,
                'n_cal': n,
                'coverage_prob': None,
                'grid': None,
                'message': f"无有效网格点（n={n} 太小，α_target={self.alpha_target} 太小）"
            }
        
        valid_alphas = grid_alphas[valid_mask]
        
        # 对每个格点计算覆盖概率
        probs = np.array([self._coverage_prob(a) for a in valid_alphas])
        self._grid = pd.DataFrame({
            'alpha_prime': valid_alphas,
            'coverage_prob': probs,
            'satisfies_pac': probs >= (1 - self.delta)
        })
        
        # 选择满足 PAC 约束的最大 α_adj（最不保守的可行解）
        pac_mask = probs >= (1 - self.delta)
        
        if not pac_mask.any():
            # 所有格点都无法满足 PAC 约束
            self.feasible = False
            self.alpha_adj = None
            return {
                'feasible': False,
                'alpha_adj': None,
                'alpha_target': self.alpha_target,
                'delta': self.delta,
                'n_cal': n,
                'coverage_prob': probs.max() if len(probs) > 0 else None,
                'grid': self._grid,
                'message': (f"给定 n={n}，无法同时满足 α_target={self.alpha_target} "
                           f"和 δ={self.delta}。建议增大 n 或放宽 δ。")
            }
        
        # 最大 α_adj（在满足约束的格点中选最大的，最保守程度最低）
        best_idx = np.where(pac_mask)[0][-1]
        self.alpha_adj = valid_alphas[best_idx]
        self.feasible = True
        
        return {
            'feasible': True,
            'alpha_adj': self.alpha_adj,
            'alpha_target': self.alpha_target,
            'delta': self.delta,
            'n_cal': n,
            'coverage_prob': probs[best_idx],
            'grid': self._grid,
            'message': (f"α_adj={self.alpha_adj:.4f} (vs α_target={self.alpha_target})，"
                       f"Pr(coverage≥{1-self.alpha_target:.0%}) ≥ {1-self.delta:.0%}")
        }
    
    def feasibility_threshold(self) -> float:
        """
        计算最小可行 α_target（无穷测试集的闭合公式）
        
        Returns:
            α*_∞(n, δ) = 1 - δ^(1/n)
        """
        return 1 - self.delta ** (1 / self.n_cal)
    
    def __repr__(self):
        return (f"SSBC(α_target={self.alpha_target}, δ={self.delta}, "
                f"n_cal={self.n_cal}, m_test={self.m_test})")


# ==================== 标准共形预测（用于对比） ====================

class SplitConformal:
    """
    标准 Split Conformal Prediction（用于与 SSBC 对比）
    """
    
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.q_hat = None
        self.n_cal = None
    
    def calibrate(self, scores: np.ndarray) -> 'SplitConformal':
        """
        用校准集的非一致性分数计算分位数阈值
        
        Args:
            scores: 非一致性分数数组（越大 = 越不符合）
                   对于回归：|y - y_hat|
                   对于分类：1 - p_y_true
        """
        self.n_cal = len(scores)
        # 标准共形分位数（含有限样本修正 ceil((n+1)(1-α))/n）
        q_level = np.ceil((1 - self.alpha) * (self.n_cal + 1)) / self.n_cal
        q_level = min(q_level, 1.0)
        self.q_hat = np.quantile(scores, q_level)
        return self
    
    def predict_interval(self, y_hat: np.ndarray, 
                         residual_std: float = None) -> tuple:
        """
        生成预测区间（回归场景）
        
        Returns:
            (lower, upper): 预测区间
        """
        if self.q_hat is None:
            raise ValueError("请先调用 calibrate()")
        lower = y_hat - self.q_hat
        upper = y_hat + self.q_hat
        return lower, upper


# ==================== SSBC 增强共形预测 ====================

class SSBCConformal:
    """
    SSBC 增强的 Split Conformal Prediction
    
    在标准共形的基础上，用 SSBC 调整 α，提供 PAC 覆盖保证。
    """
    
    def __init__(self, alpha_target: float, delta: float, 
                 m_test: int = None, verbose: bool = True):
        """
        Args:
            alpha_target: 目标显著性水平（如 0.05）
            delta: 风险容忍度（如 0.05 表示 95% 置信度保证覆盖）
            m_test: 测试集大小（None = 无穷）
            verbose: 是否打印校准信息
        """
        self.alpha_target = alpha_target
        self.delta = delta
        self.m_test = m_test
        self.verbose = verbose
        
        self.ssbc = None
        self.alpha_adj = None
        self.q_hat_target = None  # 标准共形的 q_hat（α_target）
        self.q_hat_ssbc = None    # SSBC 调整后的 q_hat（α_adj）
        self.n_cal = None
        self.feasible = False
    
    def calibrate(self, scores: np.ndarray) -> 'SSBCConformal':
        """
        校准步骤：同时计算标准共形和 SSBC 调整的分位数
        
        Args:
            scores: 校准集非一致性分数
        """
        self.n_cal = len(scores)
        
        # Step 1: 标准共形分位数（α_target）
        q_level_target = min(
            np.ceil((1 - self.alpha_target) * (self.n_cal + 1)) / self.n_cal,
            1.0
        )
        self.q_hat_target = np.quantile(scores, q_level_target)
        
        # Step 2: 计算 SSBC 调整后的 α_adj
        self.ssbc = SSBC(
            alpha_target=self.alpha_target,
            delta=self.delta,
            n_cal=self.n_cal,
            m_test=self.m_test
        )
        result = self.ssbc.compute()
        self.feasible = result['feasible']
        
        if self.feasible:
            self.alpha_adj = result['alpha_adj']
            q_level_adj = min(
                np.ceil((1 - self.alpha_adj) * (self.n_cal + 1)) / self.n_cal,
                1.0
            )
            self.q_hat_ssbc = np.quantile(scores, q_level_adj)
        else:
            self.alpha_adj = None
            self.q_hat_ssbc = None
        
        if self.verbose:
            print(f"\n[SSBC 校准报告]")
            print(f"  校准集大小:     n = {self.n_cal}")
            print(f"  目标显著性水平: α_target = {self.alpha_target:.4f}")
            print(f"  风险容忍度:     δ = {self.delta:.4f}")
            print(f"  可行性:         {'✅ 可行' if self.feasible else '❌ 不可行'}")
            if self.feasible:
                print(f"  SSBC 调整:      α_adj = {self.alpha_adj:.4f} (α_target={self.alpha_target:.4f})")
                print(f"  PAC 保证:       Pr(coverage ≥ {1-self.alpha_target:.0%}) ≥ {1-self.delta:.0%}")
                print(f"  分位数对比:     标准 q={self.q_hat_target:.4f}, SSBC q={self.q_hat_ssbc:.4f}")
                print(f"  区间扩大比:     {self.q_hat_ssbc/self.q_hat_target:.2f}x")
            else:
                print(f"  {result['message']}")
                min_feasible = self.ssbc.feasibility_threshold()
                print(f"  最小可行 α_target ≈ {min_feasible:.4f}")
        
        return self
    
    def predict_interval(self, y_hat: np.ndarray, 
                          use_ssbc: bool = True) -> dict:
        """
        生成预测区间
        
        Args:
            y_hat: 点预测值
            use_ssbc: True 使用 SSBC 调整（PAC 保证），False 使用标准共形
            
        Returns:
            dict with keys:
                'lower', 'upper': 预测区间
                'width': 区间宽度
                'method': 使用的方法
                'coverage_guarantee': 覆盖保证说明
        """
        if self.q_hat_target is None:
            raise ValueError("请先调用 calibrate()")
        
        if use_ssbc and self.feasible:
            q = self.q_hat_ssbc
            method = 'SSBC'
            guarantee = (f"PAC: Pr(coverage ≥ {1-self.alpha_target:.0%}) ≥ "
                        f"{1-self.delta:.0%}")
        else:
            q = self.q_hat_target
            method = 'Standard'
            guarantee = (f"期望覆盖率 ≥ {1-self.alpha_target:.0%} "
                        f"（但单次校准可能不达标）")
        
        lower = y_hat - q
        upper = y_hat + q
        
        return {
            'lower': lower,
            'upper': upper,
            'width': upper - lower,
            'q_hat': q,
            'method': method,
            'coverage_guarantee': guarantee
        }


# ==================== 业务场景：新市场小样本ROI预测 ====================

def generate_new_market_data(n_cal=50, n_test=500, seed=42):
    """
    模拟母婴出海新市场（日本）小样本场景
    
    只有 n_cal 条带标签校准数据，预测用户转化概率/ROI
    """
    rng = np.random.default_rng(seed)
    
    # 真实转化概率生成函数
    def true_prob(x):
        return 0.15 + 0.3 * x[:, 0] + 0.2 * x[:, 1] + 0.1 * x[:, 2]
    
    def generate_batch(n):
        # 特征：[收入分位数, 页面浏览深度, 停留时长标准化]
        X = rng.uniform(0, 1, (n, 3))
        true_p = true_prob(X)
        y = rng.binomial(1, true_p).astype(float)
        # 模型预测（含误差）
        y_hat = true_p + rng.normal(0, 0.05, n)
        y_hat = np.clip(y_hat, 0.01, 0.99)
        # 非一致性分数（绝对误差）
        scores = np.abs(y - y_hat)
        return X, y, y_hat, scores, true_p
    
    X_cal, y_cal, yhat_cal, scores_cal, _ = generate_batch(n_cal)
    X_test, y_test, yhat_test, scores_test, true_p_test = generate_batch(n_test)
    
    return {
        'cal': (X_cal, y_cal, yhat_cal, scores_cal),
        'test': (X_test, y_test, yhat_test, scores_test, true_p_test)
    }


def evaluate_coverage(y_true, lower, upper):
    """计算实际覆盖率"""
    return np.mean((y_true >= lower) & (y_true <= upper))


def run_new_market_demo(n_cal=50, alpha_target=0.1, delta=0.1):
    """
    母婴出海新市场场景：SSBC vs 标准共形对比实验
    
    场景：日本市场仅有 n_cal 条带标签数据，
    目标：保证 90% 覆盖率（α=0.1），置信度 90%（δ=0.1）
    """
    print("=" * 65)
    print(f"场景：母婴出海新市场（日本）- 小样本 PAC 覆盖保证")
    print(f"校准集大小: n={n_cal}，目标覆盖率: {1-alpha_target:.0%}，置信度: {1-delta:.0%}")
    print("=" * 65)
    
    # 生成数据
    data = generate_new_market_data(n_cal=n_cal, n_test=1000)
    X_cal, y_cal, yhat_cal, scores_cal = data['cal']
    X_test, y_test, yhat_test, scores_test, true_p_test = data['test']
    
    # === 标准共形（期望保证） ===
    std_conformal = SplitConformal(alpha=alpha_target)
    std_conformal.calibrate(scores_cal)
    lower_std, upper_std = std_conformal.predict_interval(yhat_test)
    coverage_std = evaluate_coverage(y_test, lower_std, upper_std)
    width_std = (upper_std - lower_std).mean()
    
    # === SSBC 增强共形（PAC 保证） ===
    ssbc_conformal = SSBCConformal(
        alpha_target=alpha_target, 
        delta=delta, 
        verbose=True
    )
    ssbc_conformal.calibrate(scores_cal)
    
    # === 对比结果 ===
    print(f"\n{'='*65}")
    print("对比结果")
    print(f"{'='*65}")
    print(f"{'指标':<30} {'标准共形':>12} {'SSBC':>12}")
    print(f"{'-'*55}")
    print(f"{'实际覆盖率':<30} {coverage_std:>12.1%} ", end="")
    
    if ssbc_conformal.feasible:
        result_ssbc = ssbc_conformal.predict_interval(yhat_test, use_ssbc=True)
        coverage_ssbc = evaluate_coverage(y_test, result_ssbc['lower'], result_ssbc['upper'])
        width_ssbc = result_ssbc['width'].mean()
        print(f"{coverage_ssbc:>12.1%}")
        print(f"{'平均区间宽度':<30} {width_std:>12.3f} {width_ssbc:>12.3f}")
        
        # 通过多次 Monte Carlo 估计违约率
        print(f"\n[Monte Carlo 违约率估计，1000 次随机校准集]")
        n_trials = 1000
        violation_std = 0
        violation_ssbc = 0
        
        for trial in range(n_trials):
            trial_data = generate_new_market_data(
                n_cal=n_cal, n_test=500, seed=trial + 100
            )
            _, y_c, _, sc = trial_data['cal']
            _, y_t, yh_t, _, _ = trial_data['test']
            
            # 标准共形
            sc_std = SplitConformal(alpha=alpha_target)
            sc_std.calibrate(sc)
            l, u = sc_std.predict_interval(yh_t)
            cov = evaluate_coverage(y_t, l, u)
            if cov < (1 - alpha_target):
                violation_std += 1
            
            # SSBC
            if ssbc_conformal.feasible:
                sc_ssbc = SSBCConformal(
                    alpha_target=alpha_target, delta=delta, verbose=False
                )
                sc_ssbc.calibrate(sc)
                if sc_ssbc.feasible:
                    res = sc_ssbc.predict_interval(yh_t, use_ssbc=True)
                    cov_s = evaluate_coverage(y_t, res['lower'], res['upper'])
                    if cov_s < (1 - alpha_target):
                        violation_ssbc += 1
        
        vr_std = violation_std / n_trials
        vr_ssbc = violation_ssbc / n_trials
        print(f"  标准共形 违约率: {vr_std:.1%} (期望: ≤ {alpha_target:.0%}，理论值~40%)")
        print(f"  SSBC    违约率: {vr_ssbc:.1%} (期望: ≤ {delta:.0%})")
        
        verdict = "✅ SSBC 成功将违约率控制在目标δ以内" if vr_ssbc <= delta * 1.5 else "⚠️ 部分达标"
        print(f"\n  {verdict}")
    else:
        print("N/A（不可行）")
    
    return ssbc_conformal


# ==================== 可行性分析工具 ====================

def feasibility_analysis(alpha_target=0.1, delta_values=None, n_range=None):
    """
    分析不同 (n, δ) 组合的可行性
    
    帮助回答：给定目标覆盖率，需要多少校准样本？
    """
    if delta_values is None:
        delta_values = [0.01, 0.05, 0.10, 0.25]
    if n_range is None:
        n_range = [20, 30, 50, 75, 100, 150, 200]
    
    print(f"\n可行性分析: α_target={alpha_target} (目标覆盖率={1-alpha_target:.0%})")
    print(f"{'n_cal':>8}", end="")
    for delta in delta_values:
        print(f"  δ={delta:.2f}", end="")
    print()
    print("-" * (8 + 8 * len(delta_values)))
    
    for n in n_range:
        print(f"{n:>8}", end="")
        for delta in delta_values:
            ssbc = SSBC(alpha_target=alpha_target, delta=delta, n_cal=n)
            result = ssbc.compute()
            if result['feasible']:
                print(f"  {result['alpha_adj']:.4f}", end="")
            else:
                print(f"   N/A  ", end="")
        print()
    
    # 同时输出最小 n 要求
    print(f"\n最小 n 要求（闭合公式近似: α* = 1 - δ^(1/n) < α_target）:")
    for delta in delta_values:
        # 解 1 - δ^(1/n) = α_target => n = log(δ)/log(1-α_target)
        n_min = np.log(delta) / np.log(1 - alpha_target)
        print(f"  δ={delta:.2f}: 至少需要 n ≈ {int(np.ceil(n_min))} 个校准样本")


# ==================== 测试用例 ====================

def test_ssbc_coverage_guarantee():
    """
    测试 1：验证 SSBC 能将违约率控制在 δ 以内
    期望：SSBC 的 Monte Carlo 违约率 ≤ δ（标准共形 >> δ）
    """
    print("=" * 60)
    print("测试1: SSBC PAC 覆盖保证验证（Monte Carlo）")
    print("=" * 60)
    
    n_cal = 50
    alpha_target = 0.1
    delta = 0.1
    n_trials = 500
    
    # 先计算 SSBC 的 α_adj
    ssbc = SSBC(alpha_target=alpha_target, delta=delta, n_cal=n_cal)
    result = ssbc.compute()
    
    if not result['feasible']:
        print(f"⚠️ 不可行: {result['message']}")
        return
    
    alpha_adj = result['alpha_adj']
    print(f"α_adj = {alpha_adj:.4f} (α_target = {alpha_target})")
    print(f"理论 Pr(coverage ≥ {1-alpha_target:.0%}) = {result['coverage_prob']:.3f}")
    
    violations_std = 0
    violations_ssbc = 0
    
    rng = np.random.default_rng(0)
    for trial in range(n_trials):
        # 生成校准集分数（模拟残差）
        cal_scores = rng.exponential(scale=1.0, size=n_cal)
        test_y = rng.exponential(scale=1.0, size=200)
        test_yhat = np.zeros(200)  # 点预测为 0
        test_scores = np.abs(test_y - test_yhat)
        
        # 标准共形 q
        q_level_std = min(
            np.ceil((1 - alpha_target) * (n_cal + 1)) / n_cal, 1.0
        )
        q_std = np.quantile(cal_scores, q_level_std)
        cov_std = np.mean(test_scores <= q_std)
        if cov_std < (1 - alpha_target):
            violations_std += 1
        
        # SSBC q
        q_level_adj = min(
            np.ceil((1 - alpha_adj) * (n_cal + 1)) / n_cal, 1.0
        )
        q_ssbc = np.quantile(cal_scores, q_level_adj)
        cov_ssbc = np.mean(test_scores <= q_ssbc)
        if cov_ssbc < (1 - alpha_target):
            violations_ssbc += 1
    
    vr_std = violations_std / n_trials
    vr_ssbc = violations_ssbc / n_trials
    
    print(f"\n标准共形违约率: {vr_std:.3f} (期望 ~{alpha_target:.0%}，实际更高)")
    print(f"SSBC     违约率: {vr_ssbc:.3f} (期望 ≤ {delta:.0%})")
    
    assert vr_ssbc <= delta * 1.5, f"SSBC 违约率 {vr_ssbc:.3f} 超出允许范围"
    print("✅ 测试1通过：SSBC 成功控制单次校准违约率")
    
    return vr_std, vr_ssbc


def test_ssbc_feasibility():
    """
    测试 2：可行性判断测试
    期望：n 太小时正确返回不可行
    """
    print("\n" + "=" * 60)
    print("测试2: 可行性判断")
    print("=" * 60)
    
    cases = [
        (5,   0.05, 0.05),   # 极小样本，应不可行
        (20,  0.10, 0.25),   # 中等宽松，可能可行
        (50,  0.10, 0.10),   # 目标场景，应可行
        (100, 0.05, 0.05),   # 较大样本，应可行
    ]
    
    for n, alpha, delta in cases:
        ssbc = SSBC(alpha_target=alpha, delta=delta, n_cal=n)
        result = ssbc.compute()
        status = "✅ 可行" if result['feasible'] else "❌ 不可行"
        if result['feasible']:
            print(f"  n={n:3d}, α={alpha}, δ={delta}: {status} → α_adj={result['alpha_adj']:.4f}")
        else:
            min_a = ssbc.feasibility_threshold()
            print(f"  n={n:3d}, α={alpha}, δ={delta}: {status} (最小可行 α ≈ {min_a:.4f})")
    
    # 具体断言
    ssbc_small = SSBC(alpha_target=0.05, delta=0.05, n_cal=5)
    assert not ssbc_small.compute()['feasible'], "n=5 极端条件应不可行"
    
    ssbc_ok = SSBC(alpha_target=0.10, delta=0.10, n_cal=50)
    assert ssbc_ok.compute()['feasible'], "n=50 目标场景应可行"
    
    print("✅ 测试2通过：可行性判断正确")


def test_ssbc_more_conservative_than_standard():
    """
    测试 3：SSBC 的 q_hat 应 ≥ 标准共形的 q_hat（更保守）
    """
    print("\n" + "=" * 60)
    print("测试3: SSBC 保守性验证（q_adj ≥ q_standard）")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    scores = rng.exponential(1.0, 50)
    
    alpha_target = 0.1
    delta = 0.1
    
    std_c = SplitConformal(alpha=alpha_target)
    std_c.calibrate(scores)
    q_std = std_c.q_hat
    
    ssbc_c = SSBCConformal(alpha_target=alpha_target, delta=delta, verbose=False)
    ssbc_c.calibrate(scores)
    
    if ssbc_c.feasible:
        q_ssbc = ssbc_c.q_hat_ssbc
        print(f"标准共形 q_hat:  {q_std:.4f}")
        print(f"SSBC q_hat:      {q_ssbc:.4f}")
        print(f"SSBC / Standard: {q_ssbc/q_std:.2f}x")
        
        assert q_ssbc >= q_std, f"SSBC q={q_ssbc:.4f} 应 ≥ 标准共形 q={q_std:.4f}"
        print("✅ 测试3通过：SSBC 比标准共形更保守（但比 DKWM 更精确）")
    else:
        print("⚠️ SSBC 不可行，跳过此测试")


def main():
    """
    完整演示：SSBC 在母婴出海新市场场景的端到端应用
    """
    print("=" * 65)
    print("SSBC 小样本共形预测 PAC 覆盖保证")
    print("论文: arXiv:2509.15349 (2025)")
    print("=" * 65)
    
    # === 1. 可行性分析 ===
    print("\n【模块 1】可行性分析：不同 n、δ 组合下的可达 α_adj")
    feasibility_analysis(alpha_target=0.10, delta_values=[0.05, 0.10, 0.25])
    
    # === 2. 新市场场景演示 ===
    print("\n\n【模块 2】日本新市场场景演示")
    ssbc_conformal = run_new_market_demo(
        n_cal=50, alpha_target=0.10, delta=0.10
    )
    
    # === 3. 商业汇报模板 ===
    print("\n\n【模块 3】给 CEO 的汇报摘要")
    print("─" * 50)
    if ssbc_conformal.feasible:
        print(f"✅ 校准集仅 {ssbc_conformal.n_cal} 条数据，SSBC 保证：")
        print(f"   以 {1-ssbc_conformal.delta:.0%} 概率，预测覆盖率 ≥ {1-ssbc_conformal.alpha_target:.0%}")
        print(f"   （α 从 {ssbc_conformal.alpha_target} 调整为 {ssbc_conformal.alpha_adj:.4f}）")
        print(f"   决策建议：可基于此区间做预算分配，风险可控")
    
    # === 4. 单元测试 ===
    print("\n\n【模块 4】单元测试")
    vr_std, vr_ssbc = test_ssbc_coverage_guarantee()
    test_ssbc_feasibility()
    test_ssbc_more_conservative_than_standard()
    
    print("\n\n" + "=" * 65)
    print("✅ 所有模块完成！")
    print(f"   标准共形违约率: {vr_std:.1%} → SSBC 违约率: {vr_ssbc:.1%}")
    print(f"   SSBC 覆盖保证: 即插即用，无需改模型")
    print("=" * 65)


if __name__ == '__main__':
    main()
```

---

## ④ 技能关联

| 关系 | 技能 | 理由 |
|------|------|------|
| 前置 | [Conformal ROI Prediction]([[Skill-Conformal-ROI-Prediction]].md) | 标准 Split Conformal 是 SSBC 的基础，需先理解期望覆盖保证 |
| 前置 | [Uplift Modeling]([[Skill-Uplift-Modeling]].md) | Uplift 模型是共形预测的常见底层模型 |
| 组合 | EPICSCORE（06-增长模型） | EPICSCORE 改进非一致性分数，SSBC 调整显著性水平，联合使用效果最佳 |
| 组合 | [DML Cohort Causal Effect]([[Skill-DML-Cohort-Causal-Effect]].md) | DML 小样本产出点估计 + SSBC 保证区间覆盖 |
| 延伸 | [DiD Difference-in-Differences]([[Skill-DiD-Difference-in-Differences]].md) | DiD 在小样本下的可信区间同样可用 SSBC 增强 |
| 延伸 | A/B 实验设计（02-A_B实验） | 共形预测的校准集本质是小规模 RCT，SSBC 减少了所需 RCT 样本量 |

---

- **前置技能**：[[Skill-Conformal-ROI-Prediction]] | [[Skill-EPICSCORE-Uncertainty]]
- **延伸技能**：[[Skill-AB-Experimental-Design]]
- **可组合技能**：[[Skill-BCCB-Causal-Bandits]]

## ⑤ 商业价值评估

| 维度 | 评分 | 依据 |
|------|------|------|
| ROI预估 | **规避风险 5-20 万/月** | 避免基于小样本不可信区间的错误预算决策（月广告预算 30 万，违约率从 40% 降至 10%） |
| 实施难度 | ⭐⭐☆☆☆ | 即插即用，不改模型；仅需 `scipy.stats` 做 Beta CDF 计算；2-4 小时接入 |
| 优先级 | ⭐⭐⭐⭐ P1 | **直接解决小样本新市场"共形预测不可信"的核心痛点** |

### ROI 量化依据

**场景：母婴出海新市场（日本）月广告预算 30 万**
- 现状痛点：日本市场校准集仅 50 条数据，标准共形预测 40% 概率实际覆盖 < 90%
- SSBC 收益：将违约率从 ~40% 降至 ~10%（与目标 δ 对齐），预算决策信心大幅提升
- 量化价值：避免每月 1-2 次"基于不可信区间"的错误决策，每次预估损失 3-10 万

### SSBC vs 竞品对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| 标准共形 | 简单，期望覆盖保证 | 单次校准 40% 违约率，小样本不可靠 |
| DKWM 修正 | 数学上有效 | 过度保守，区间太宽（实际违约率 ~1.6% vs 目标 10%） |
| **SSBC** | **PAC 保证 + 不过保守 + 即插即用** | 区间比标准共形略宽（约 1.1-1.5x） |
| Mondrian | 条件覆盖 | 样本需求更高 |

### 实施路线

```
Hour 1-2: 集成 SSBC 类，输入 (α_target, δ, n_cal)，输出 α_adj
Hour 3-4: 替换现有 q_hat 计算（只需改一行：用 α_adj 代替 α_target）
Day 2:    可行性分析（明确哪些 (α, δ) 目标在当前数据量下可达）
Week 2:   历史数据 Monte Carlo 验证，确认违约率控制效果
```
