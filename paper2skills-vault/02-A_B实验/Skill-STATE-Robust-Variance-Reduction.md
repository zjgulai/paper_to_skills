---
title: STATE — 重尾指标鲁棒 A/B 方差减少：Student-t 回归调整（-70% 方差）
doc_type: knowledge
module: 02-A_B实验
topic: state-robust-variance-reduction-heavy-tail
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: STATE — 重尾指标鲁棒 A/B 方差减少

> **领域**: 02-A_B实验 | **类型**: 综合萃取 | **来源**: arXiv:2407.16337（美团，2024）

---

## ① 算法原理

**核心问题**：电商 GMV / 订单量等指标天然重尾——极少数大客户的超大订单把方差撑得很高。CUPED 假设残差服从高斯分布，重尾数据下高斯假设失效，方差缩减效率大打折扣；CUPAC/MLRATE 用机器学习拟合协变量改进了均值预测，但同样没处理残差的非高斯性。

**STATE 的解法**：把回归残差 $\varepsilon = Y - f(X)$ 建模为 **Student's t-分布** 而非高斯分布。t-分布自由度 $\nu$ 控制尾部厚度：$\nu$ 小则尾部重（容纳极端值），$\nu \to \infty$ 退化为高斯。t-分布对离群值的下降速率远比高斯慢，使得极端订单不再"撑爆"方差估计。

**求解框架**：变分 EM（Expectation-Maximization）：
- E 步：计算每个样本的隐变量权重 $w_i = (\nu + 1) / (\nu + r_i^2/\sigma^2)$，$r_i$ 为残差
- M 步：加权最小二乘更新回归系数；在对数似然下更新 $\nu$（自由度）和 $\sigma$（尺度）
- 迭代直至参数收敛

**两类指标处理**：
- **计数指标**（订单量）：残差建模为 t-分布，直接加权回归
- **比率指标**（GMV/用户）：先 delta 展开为线性化形式，再用 STATE 估计方差

**方差缩减量化**：美团真实 A/A 测试，STATE 对订单量指标方差减少 **70.5%**（vs CUPED），对 GMV **80.7%**；相比 CUPAC/MLRATE 额外减少 **36%+**。同等统计功效下实验时长缩短约 **50%**。

---

## ② 母婴出海应用案例

**场景一：母婴新品 Listing AB 测试（GMV 重尾问题）**

- **业务问题**：新品奶粉上架后测试两种 Listing 方案（主图 A vs 主图 B），以 GMV 作为主指标。少数大客户（批量采购/月子中心）产生 10-50 倍于普通用户的订单，使 GMV 方差极高。传统 CUPED 需跑 3 周才显著；
- **数据要求**：实验期 GMV（$Y$）+ 实验前 14 天 GMV/浏览/加购等协变量（$X$），用户级别数据
- **STATE 做法**：以 XGBoost 拟合 $\hat{Y} = f(X)$，残差 $\varepsilon$ 用 EM 估 t 分布参数（典型 $\hat{\nu} \approx 3.5$），得到方差缩减后的调整指标
- **预期产出**：方差缩减 70%+，同等功效下实验从 3 周缩短至 **1.5 周**即可得出显著结论
- **业务价值**：母婴新品 Listing 迭代周期减半，每个实验节省约 1.5 周，1 年可多跑约 17 轮实验

**场景二：WF-B 广告竞价策略测试（点击/订单量重尾）**

- **业务问题**：测试 Google Shopping 出价策略 A vs B，以订单量和 ROAS 为指标。流量高峰期（母婴促销节）产生巨量短期爆单，订单量分布严重右偏
- **数据要求**：广告账户级别实验，用户曝光→点击→订单链路数据；实验前 7 天点击/订单作为协变量
- **STATE 做法**：对订单量用计数型 STATE（t 残差），对 ROAS 用比率型 STATE（delta 展开后 t 残差），EM 估参后得到功效更高的检验统计量
- **预期产出**：检测 +5% ROAS 提升所需时间从 4 周→**2 周**，避免广告策略在次优方案上空转 2 周
- **业务价值**：广告 ROAS 优化加速，每月可多验证 1-2 个竞价假设，直接提升广告 ROI 迭代速度

---

## ③ 代码模板

```python
"""
STATE Variance Reduction — Student's t Regression Adjustment
arXiv:2407.16337 | 美团 2024
纯标准库 + numpy/scipy 实现
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from scipy import stats


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class HeavyTailMetrics:
    """实验指标数据容器"""
    metric_values: np.ndarray        # 实验期指标 Y（GMV / 订单量）
    pre_exp_covariates: np.ndarray   # 实验前协变量 X（可多列）
    treatment: np.ndarray            # 处理标记：1=实验组，0=对照组

    def __post_init__(self):
        n = len(self.metric_values)
        assert len(self.treatment) == n, "treatment length mismatch"
        if self.pre_exp_covariates.ndim == 1:
            self.pre_exp_covariates = self.pre_exp_covariates.reshape(-1, 1)
        assert len(self.pre_exp_covariates) == n, "covariate length mismatch"

    @property
    def treatment_idx(self) -> np.ndarray:
        return self.treatment == 1

    @property
    def control_idx(self) -> np.ndarray:
        return self.treatment == 0


@dataclass
class ATEResult:
    """ATE 估计结果"""
    ate: float                       # 平均处理效应
    variance: float                  # 估计量方差
    std_error: float                 # 标准误
    p_value: float                   # 双侧 p-value
    ci_lower: float                  # 95% 置信区间下界
    ci_upper: float                  # 95% 置信区间上界
    t_df: Optional[float] = None     # t 分布自由度（STATE 专属）
    t_scale: Optional[float] = None  # t 分布尺度（STATE 专属）
    variance_reduction_vs_raw: float = 0.0  # vs 原始方差的缩减率

    def __str__(self) -> str:
        sig = "✓ 显著" if self.p_value < 0.05 else "✗ 不显著"
        return (
            f"ATE={self.ate:.4f}  SE={self.std_error:.4f}  "
            f"p={self.p_value:.4f}  {sig}\n"
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"方差缩减={self.variance_reduction_vs_raw:.1%}"
            + (f"  t_df={self.t_df:.2f}  t_scale={self.t_scale:.4f}" if self.t_df else "")
        )


# ─── CUPED 基线 ────────────────────────────────────────────────────────────────

class CUPEDEstimator:
    """
    标准 CUPED（Controlled-experiment Using Pre-Experiment Data）
    Y_adj = Y - theta * (X - mean(X))，theta = Cov(Y,X) / Var(X)
    """

    def __init__(self):
        self.theta_: Optional[np.ndarray] = None
        self.x_mean_: Optional[np.ndarray] = None

    def _linear_adjust(self, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """OLS 投影，返回 Y 的残差（调整后指标）"""
        X_aug = np.column_stack([np.ones(len(X)), X])
        coef, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
        return Y - X_aug @ coef

    def estimate_ate(self, data: HeavyTailMetrics) -> ATEResult:
        Y = data.metric_values
        X = data.pre_exp_covariates
        T = data.treatment

        Y_adj = self._linear_adjust(Y, X)
        raw_var = float(np.var(Y[T == 1]) / T.sum() + np.var(Y[T == 0]) / (1 - T).sum())

        Yt_adj = Y_adj[data.treatment_idx]
        Yc_adj = Y_adj[data.control_idx]

        ate = float(np.mean(Yt_adj) - np.mean(Yc_adj))
        var_t = float(np.var(Yt_adj, ddof=1) / len(Yt_adj))
        var_c = float(np.var(Yc_adj, ddof=1) / len(Yc_adj))
        var_ate = var_t + var_c
        se = math.sqrt(var_ate)

        z = ate / se
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        var_reduction = 1.0 - var_ate / raw_var if raw_var > 0 else 0.0

        return ATEResult(
            ate=ate, variance=var_ate, std_error=se,
            p_value=p_value, ci_lower=ci_lower, ci_upper=ci_upper,
            variance_reduction_vs_raw=var_reduction,
        )


# ─── STATE 估计器 ──────────────────────────────────────────────────────────────

class STATEEstimator:
    """
    STATE: Student's t Regression Adjustment for ATE Estimation
    通过 EM 估计残差的 t 分布参数，加权回归后计算 ATE
    arXiv:2407.16337
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.df_: Optional[float] = None      # 自由度 nu
        self.scale_: Optional[float] = None   # 尺度 sigma

    # ── EM 估计 t 分布参数 ────────────────────────────────────────────────────

    def fit_t_distribution(
        self, residuals: np.ndarray
    ) -> Tuple[float, float]:
        """
        变分 EM 估计 Student-t 分布参数 (df, scale)
        E步：计算隐变量权重 w_i = (nu+1)/(nu + r_i^2/sigma^2)
        M步：加权 MLE 更新 nu, sigma
        返回：(df, scale)
        """
        r = residuals - np.mean(residuals)  # 中心化
        n = len(r)

        # 初始化：矩估计
        sigma = float(np.std(r))
        nu = 10.0  # 初始自由度

        for iteration in range(self.max_iter):
            # E步：期望权重（t 分布的辅助变量）
            weights = (nu + 1) / (nu + (r / sigma) ** 2)

            # M步：更新 sigma²（加权方差）
            sigma_new = float(np.sqrt(np.sum(weights * r ** 2) / n))

            # M步：更新 nu（数值优化，用 digamma 梯度）
            nu_new = self._update_nu(weights, nu)

            # 收敛检验
            if abs(sigma_new - sigma) < self.tol and abs(nu_new - nu) < self.tol:
                sigma, nu = sigma_new, nu_new
                break
            sigma, nu = sigma_new, nu_new

        self.df_ = max(nu, 2.01)   # nu > 2 保证方差有限
        self.scale_ = sigma
        return self.df_, self.scale_

    @staticmethod
    def _update_nu(weights: np.ndarray, nu_prev: float) -> float:
        """用 Newton-Raphson 更新自由度 nu"""
        from math import lgamma
        n = len(weights)

        def neg_log_likelihood(nu: float) -> float:
            if nu <= 2:
                return 1e10
            log_w_sum = float(np.sum(np.log(weights) - weights))
            # 近似对数似然关于 nu 的梯度
            return -(n * (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                         - 0.5 * math.log(nu)) + 0.5 * (nu + 1) * log_w_sum)

        # 简单 grid search（nu 取 1.5-50）
        candidates = np.arange(2.1, 50.1, 0.5)
        losses = [neg_log_likelihood(float(c)) for c in candidates]
        return float(candidates[int(np.argmin(losses))])

    # ── 加权回归 ─────────────────────────────────────────────────────────────

    def _weighted_linear_adjust(
        self, Y: np.ndarray, X: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """加权 OLS 投影，返回残差"""
        X_aug = np.column_stack([np.ones(len(X)), X])
        W = np.diag(weights)
        # WLS: (X'WX)^{-1} X'WY
        XtW = X_aug.T @ W
        coef = np.linalg.lstsq(XtW @ X_aug, XtW @ Y, rcond=None)[0]
        return Y - X_aug @ coef

    def estimate_ate(self, data: HeavyTailMetrics) -> ATEResult:
        """
        STATE ATE 估计流程：
        1. 初步 OLS 残差
        2. EM 估计 t 分布参数
        3. 加权 OLS 重新调整
        4. 方差估计使用 t 加权协方差
        """
        Y = data.metric_values
        X = data.pre_exp_covariates
        T = data.treatment

        # 步骤1：初步 OLS 得到残差
        X_aug = np.column_stack([np.ones(len(X)), X])
        coef0, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
        residuals0 = Y - X_aug @ coef0

        # 步骤2：EM 估计 t 参数
        nu, sigma = self.fit_t_distribution(residuals0)

        # 步骤3：t 加权 OLS
        weights = (nu + 1) / (nu + (residuals0 / sigma) ** 2)
        Y_adj = self._weighted_linear_adjust(Y, X, weights)

        # 步骤4：ATE 估计
        raw_var = float(np.var(Y[T == 1]) / T.sum() + np.var(Y[T == 0]) / (1 - T).sum())
        Yt_adj = Y_adj[data.treatment_idx]
        Yc_adj = Y_adj[data.control_idx]

        ate = float(np.mean(Yt_adj) - np.mean(Yc_adj))

        # t 加权方差估计（更鲁棒）
        wt = weights[data.treatment_idx]
        wc = weights[data.control_idx]
        var_t = float(np.sum(wt * (Yt_adj - np.mean(Yt_adj)) ** 2) / (np.sum(wt) ** 2 / len(wt)))
        var_c = float(np.sum(wc * (Yc_adj - np.mean(Yc_adj)) ** 2) / (np.sum(wc) ** 2 / len(wc)))
        var_ate = var_t + var_c
        se = math.sqrt(var_ate)

        # 用 t 分布计算 p-value（自由度 = nu）
        t_stat = ate / se
        df_test = min(len(Yt_adj), len(Yc_adj)) - 1
        p_value = float(2 * stats.t.sf(abs(t_stat), df=df_test))
        ci_lower = ate - stats.t.ppf(0.975, df=df_test) * se
        ci_upper = ate + stats.t.ppf(0.975, df=df_test) * se
        var_reduction = 1.0 - var_ate / raw_var if raw_var > 0 else 0.0

        return ATEResult(
            ate=ate, variance=var_ate, std_error=se,
            p_value=p_value, ci_lower=ci_lower, ci_upper=ci_upper,
            t_df=nu, t_scale=sigma,
            variance_reduction_vs_raw=var_reduction,
        )

    def variance_reduction_vs_cuped(self, data: HeavyTailMetrics) -> float:
        """STATE 相对 CUPED 的额外方差缩减率"""
        cuped = CUPEDEstimator()
        cuped_result = cuped.estimate_ate(data)
        state_result = self.estimate_ate(data)
        if cuped_result.variance <= 0:
            return 0.0
        return 1.0 - state_result.variance / cuped_result.variance


# ─── 样本量计算 ────────────────────────────────────────────────────────────────

class ABTestPowerCalculator:
    """
    基于 STATE 方差缩减的样本量计算
    n_state = n_traditional * (1 - variance_reduction)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        power: float = 0.80,
    ):
        self.alpha = alpha
        self.power = power

    def _z_scores(self) -> Tuple[float, float]:
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.power)
        return z_alpha, z_beta

    def sample_size_traditional(self, sigma: float, mde: float) -> int:
        """传统 t 检验样本量（每组）"""
        z_alpha, z_beta = self._z_scores()
        n = 2 * ((z_alpha + z_beta) * sigma / mde) ** 2
        return math.ceil(n)

    def sample_size_state(
        self,
        sigma: float,
        mde: float,
        variance_reduction: float,
    ) -> int:
        """STATE 方差缩减后的所需样本量（每组）"""
        n_trad = self.sample_size_traditional(sigma, mde)
        n_state = math.ceil(n_trad * (1 - variance_reduction))
        return max(n_state, 10)

    def experiment_days(
        self,
        daily_users: int,
        sigma: float,
        mde: float,
        variance_reduction: float = 0.70,
    ) -> dict:
        """估算实验天数"""
        n_trad = self.sample_size_traditional(sigma, mde)
        n_state = self.sample_size_state(sigma, mde, variance_reduction)
        days_trad = math.ceil(n_trad * 2 / daily_users)
        days_state = math.ceil(n_state * 2 / daily_users)
        return {
            "n_per_group_traditional": n_trad,
            "n_per_group_state": n_state,
            "days_traditional": days_trad,
            "days_state": days_state,
            "days_saved": days_trad - days_state,
            "speedup_ratio": round(days_trad / max(days_state, 1), 2),
        }


# ─── 测试 ──────────────────────────────────────────────────────────────────────

def _simulate_heavy_tail_gmv(n: int, seed: int = 42) -> HeavyTailMetrics:
    """
    模拟重尾 GMV 数据：
    - 90% 普通用户：GMV ~ Normal(200, 50)
    - 10% 大客户：GMV ~ Normal(2000, 500)（批量采购/月子中心）
    """
    rng = np.random.default_rng(seed)
    # 实验前协变量（过去 14 天 GMV）
    normal_pre = rng.normal(180, 40, int(n * 0.9))
    heavy_pre = rng.normal(1800, 400, int(n * 0.1))
    X = np.concatenate([normal_pre, heavy_pre])

    # 实验期指标（含真实效应 +5%）
    treatment = rng.integers(0, 2, n)
    effect = 10.0  # 真实 ATE = 10 元 GMV

    normal_Y = rng.normal(200, 50, int(n * 0.9))
    heavy_Y = rng.normal(2000, 500, int(n * 0.1))
    Y_base = np.concatenate([normal_Y, heavy_Y])
    Y = Y_base + effect * treatment[:n]  # 简化：effect for all treated

    # 调整长度一致
    min_len = min(len(X), len(Y), n)
    return HeavyTailMetrics(
        metric_values=Y[:min_len],
        pre_exp_covariates=X[:min_len],
        treatment=treatment[:min_len],
    )


def run_test():
    print("=" * 60)
    print("STATE 重尾指标方差缩减测试")
    print("=" * 60)

    # 生成重尾 GMV 数据
    data = _simulate_heavy_tail_gmv(n=2000)
    print(f"\n数据规模: {len(data.metric_values)} 用户")
    print(f"GMV 均值: {np.mean(data.metric_values):.1f}，标准差: {np.std(data.metric_values):.1f}")
    print(f"最大 GMV: {np.max(data.metric_values):.0f}（重尾特征）")

    # CUPED 基线
    cuped = CUPEDEstimator()
    cuped_result = cuped.estimate_ate(data)
    print(f"\n[CUPED]")
    print(cuped_result)

    # STATE
    state = STATEEstimator()
    state_result = state.estimate_ate(data)
    print(f"\n[STATE]")
    print(state_result)

    # 对比
    extra_reduction = state.variance_reduction_vs_cuped(data)
    print(f"\n[对比]")
    print(f"STATE 相对 CUPED 额外方差缩减: {extra_reduction:.1%}")

    # 样本量计算
    calc = ABTestPowerCalculator()
    sigma = float(np.std(data.metric_values))
    mde = 10.0  # 最小可检测效应：10 元 GMV
    result = calc.experiment_days(
        daily_users=5000,
        sigma=sigma,
        mde=mde,
        variance_reduction=state_result.variance_reduction_vs_raw,
    )
    print(f"\n[样本量 & 实验天数]")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # 断言基本正确性
    assert state_result.variance <= cuped_result.variance * 1.1, \
        "STATE 方差应不大于 CUPED 方差（允许10%误差）"
    assert result["days_state"] <= result["days_traditional"], \
        "STATE 实验天数应不超过传统方法"
    print("\n[✓] STATE 测试全部通过")


if __name__ == "__main__":
    run_test()
```

---

## ④ 技能关联

- **前置**：[[Skill-AB-Experimental-Design]] / [[Skill-Power-Analysis-Sample-Size]] / [[Skill-AB-Test-Result-Interpretation]]
- **延伸**：[[Skill-Switchback-Experiment-Design]] / [[Skill-BCCB-Causal-Bandits]]
- **可组合**：[[Skill-ROAS-Budget-Optimization]] / [[Skill-Guardrailed-Uplift-Targeting]] / [[Skill-Model-Evaluation-Metrics]]

---


- **跨域关联**：[[Skill-DML-Cohort-Causal-Effect]] / [[Skill-Causal-Time-Series-Forecasting-GCF]]
- **跨域关联**：[[Skill-KG-Auto-Construction-Agent-Driven]]

## ⑤ 商业价值

- **ROI**：实验时长减半（3 周→1.5 周），母婴电商每实验节省 1.5 周迭代时间；年化可多跑 ~17 轮实验，隐性价值 **30-60 万元**（每轮实验加速决策带来的优化收益）
- **实施难度**：⭐⭐⭐☆☆（需理解 EM 算法，但有成熟代码模板）
- **优先级**：⭐⭐⭐⭐⭐（重尾指标是电商 A/B 实验的普遍痛点，直接缩短实验周期）
- **评估依据**：美团真实数据验证 70.5%/80.7% 方差减少；母婴 GMV 指标天然重尾（大客户聚合效应），效果复现概率高
